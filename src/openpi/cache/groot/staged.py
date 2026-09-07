"""Split one GR00T N1.5 forward pass into three cacheable stages.

Where the cut is
----------------
Upstream runs the whole vision-language half in a single call. This module
reproduces its first six statements and stops at the point where the image
embeddings have been scattered into the language sequence but the language
model has not yet seen them::

    input_embeds = language_model.get_input_embeddings()(input_ids)
    vit_embeds   = extract_feature(pixel_values)
    input_embeds[input_ids == image_token_index] = vit_embeds
    input_embeds = input_embeds.reshape(B, N, C)      # <- stage 1 ends here
    outputs = language_model(inputs_embeds=input_embeds, ...)   # <- stage 2

Stage 1's output is therefore both the cache key source and the *only* input
stage 2 needs, which is what makes the cut clean. Cutting earlier — at the
vision tower's output — would force stage 2 to redo the language embedding
and the scatter, so neither the split nor its timings would be honest.

The second cut is between the language model and the action head. Stage 2
(`run_stage2_llm`) stops at the backbone features; stage 3 (`run_stage3`) is
the flow-matching loop, and `run_stage3_from` resumes that loop from a cached
snapshot x_t — the WARM_START path. Upstream's `get_action` draws its noise
internally and runs the whole loop as one call, so the resumable path is a
transcription of that loop (`denoise_loop`) with the noise hoisted out, pinned
by `UPSTREAM_ACTION_HEAD_SHA256`. The full MISS path still calls upstream's
`get_action` itself: the transcription is only ever used where upstream has no
entry point, and its equivalence to upstream is a tested gate, not an assumption.
The loop's step count is a runtime property of the served policy
(`action_head.num_inference_timesteps`), so the schedule every snapshot is keyed
under is derived from that live value (`live_schedule`) and never restated.

Why the upstream statements are copied rather than called
---------------------------------------------------------
The model class is loaded through `trust_remote_code`, so it lives in the
HuggingFace dynamic-module cache, outside this repository and outside its
version control. Editing it there would be invisible to review and to CI, and
monkeypatching its `forward` is a global side effect. Copying six statements
and pinning a hash of the original is the auditable option:
`UPSTREAM_FORWARD_SHA256` fails loudly the moment upstream changes, instead of
letting the copy drift into silently wrong numbers.

Deliberately not copied: upstream's `try/except` fallback around the scatter,
which re-pads when the image-token count disagrees with the vision output.
`run_stage1` asserts the exact-match precondition instead, so a mismatch is an
error here rather than a silent reshape.

The autocast contract
---------------------
`session()` owns the inference/autocast context and both stages assert they
are inside one. This is not ceremony: `LayerNorm` is on autocast's fp32 list,
so the action head's `vlln` computes in fp32 under autocast and in bf16
without it — a max|delta| of ~1.4e-2, which then feeds four Euler steps. A
collector that forgets the context would produce keys that disagree with the
online path everywhere, with nothing raising.

The context deliberately covers *only* the two forwards. Tensors that outlive
the call — cache keys, action chunks — must be created outside it, because a
tensor produced inside `inference_mode` stays an inference tensor even after
`.cpu().float()`, and in-place writes to it later raise. Building keys outside
also means pooling happens in fp32, matching the offline artifact path.

Coupling map:
  DEPENDS ON:  a GR00T_N1_5-shaped model (duck-typed), SystemTimer,
               openpi.cache.types.DenoiseSchedule
  CONSUMED BY: openpi.cache.groot.interceptor, the HDF5 collector,
               exp/robocasa365/bench_groot_stages.py (imports the loop)
  IF CHANGED:  G0-C three-stage equivalence (full and resumed) must be re-run
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn.functional as F

from openpi.cache.timing import SystemTimer
from openpi.cache.types import DenoiseSchedule, groot_n15_schedule

logger = logging.getLogger(__name__)

# id(eagle module) -> {"fn": compiled extract_feature, "checked": bool}.
# One compiled artifact (and one equivalence check) per model per process —
# see GrootStagedRunner.__init__ (compile_vision).
_COMPILED_VISION_REGISTRY: dict[int, dict[str, Any]] = {}

# ------------------------------------------------------------------
# Upstream contract
# ------------------------------------------------------------------

# sha256 of inspect.getsource(Eagle2_5_VLForConditionalGeneration.forward) for
# the n1.5-release worktree at 4af2b62 (repo copy and HuggingFace dynamic-module
# cache copy verified byte-identical, 2026-08-17).
UPSTREAM_FORWARD_SHA256 = (
    "5c58c1d2d2a9893d3f9dd1790e0e2161e94be375b4e826e46f2b26384f55c056"
)

# sha256 of the upstream action-head source file
# (gr00t/model/action_head/flow_matching_action_head.py) at the same n1.5
# worktree. `denoise_loop` below is a transcription of its `get_action`; the
# file is hashed rather than the method because the head is loaded through
# trust_remote_code and its dynamic-module copy is what actually runs.
UPSTREAM_ACTION_HEAD_SHA256 = (
    "8a8e6cf7ec63e2a335559990c4ab62bbb81e487d82ea4a969f452a93e0dbdd69"
)

# Exactly the keys the RoboCasa transform chain produces, minus `image_sizes`
# which upstream drops. Checked as a set rather than "image_flags is None"
# because that assertion is vacuous: it holds today and would keep holding if
# a future processor added a key our copy silently ignores.
_EAGLE_INPUT_KEYS = frozenset({"input_ids", "attention_mask", "pixel_values"})

_TOKENS_PER_IMAGE = 256


# ------------------------------------------------------------------
# Inter-stage data structures
# ------------------------------------------------------------------


@dataclass
class GrootStage1Output:
    """Everything stage 2 needs, and everything the cache key is cut from.

    ``input_embeds`` is the scattered vision+language sequence at the language
    model's doorstep. Under ``session()`` it is bf16 and an inference tensor;
    both facts matter downstream, so it is passed on untouched.

    There is deliberately no ``.to(device)`` helper. Any `.contiguous()` /
    `.clone()` / `.to()` between the stages can change tensor strides and so
    the kernel the attention implementation picks, which changes reduction
    order and breaks bit-exactness against the unsplit forward.
    """

    input_embeds: torch.Tensor
    """[B, N, C] — C is 2048 for the RoboCasa checkpoint."""

    attention_mask: torch.Tensor
    """[B, N]"""

    image_token_mask: torch.Tensor
    """[B, N] bool — where the vision embeddings were scattered."""

    action_inputs: Any
    """Opaque BatchFeature from ``model.prepare_input``, forwarded verbatim."""

    @property
    def state(self) -> torch.Tensor:
        """[B, T, state_dim] normalised state, already cast to the model dtype."""
        return self.action_inputs["state"]

    @property
    def state_mask(self) -> torch.Tensor:
        """[B, T, state_dim] bool — which state dimensions are real, not padding."""
        return self.action_inputs["state_mask"]


@dataclass
class GrootStage2Output:
    """Language-model output, plus the action chunk when the head already ran.

    ``backbone_features`` / ``attention_mask`` are the raw conditioning the
    action head consumes; every stage-3 call rebuilds the head's input mapping
    from them because ``process_backbone_output`` normalises in place.
    """

    backbone_features: torch.Tensor
    """[B, N, C] ``eagle_linear(hidden_states[select_layer])``, un-normalised."""

    attention_mask: torch.Tensor
    """[B, N]"""

    action_inputs: Any = None
    """Opaque BatchFeature from stage 1 (state, state_mask, embodiment_id), forwarded."""

    action_pred: Optional[torch.Tensor] = None
    """[B, action_horizon, action_dim]; ``None`` after ``run_stage2_llm`` alone."""


@dataclass
class GrootStage3Output:
    """Action chunk in normalised space, before the policy's inverse transform."""

    action_pred: torch.Tensor
    """[B, action_horizon, action_dim]"""

    start_t: Optional[float]
    """Snapshot timestep the loop resumed from; ``None`` for a full run."""

    steps_run: int
    """Euler steps actually executed (``num_steps`` for a full run)."""


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


def _autocast_dtype(device_type: str) -> Optional[torch.dtype]:
    """Current autocast dtype for a device type, across torch versions.

    The device type has to be passed explicitly: the no-argument spellings of
    both queries answer for CUDA regardless of where the model lives, so a
    CPU-hosted model would look like it had no autocast at all.
    """
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:  # torch < 2.4: per-device query did not exist
        enabled = (
            torch.is_autocast_enabled()
            if device_type == "cuda"
            else torch.is_autocast_cpu_enabled()
        )
    if not enabled:
        return None
    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):  # torch < 2.4 spelling
        if device_type == "cuda":
            return torch.get_autocast_gpu_dtype()
        return torch.get_autocast_cpu_dtype()


class GrootStagedRunner:
    """Run a GR00T model in two halves, with the cut described in the module docstring.

    Args:
        model: a ``GR00T_N1_5``-shaped object. Duck-typed on purpose so tests
            can drive the whole split with a stub; the production instance
            comes from ``Gr00tPolicy.model`` inside the GR00T island.
        timer: shared ``SystemTimer``. The runner owns the three stage probes
            (``stage1_vision`` / ``stage2_llm`` / ``stage2_action``) because
            two of them start and end inside ``run_stage2``, where no caller
            can reach. ``None`` installs a disabled timer, which is how the
            server's start-up handshake avoids polluting the first episode's
            statistics.
        verify_upstream: check the pinned source hash. Leave on.

    Raises:
        RuntimeError: if the model is in training mode, if the truncated layer
            stack disagrees with ``select_layer``, or if upstream's forward has
            changed.
    """

    def __init__(
        self,
        model: Any,
        *,
        timer: Optional[SystemTimer] = None,
        verify_upstream: bool = True,
        compile_vision: bool = False,
    ) -> None:
        self._model = model
        self._backbone = model.backbone
        self._eagle = self._backbone.eagle_model

        # Upstream's EagleBackbone.forward calls set_frozen_modules_to_eval_mode()
        # before anything else. We do not reproduce that call, so its guarantee
        # has to be established here instead of assumed.
        if getattr(model, "training", False):
            raise RuntimeError(
                "GrootStagedRunner requires an eval-mode model: the upstream "
                "backbone re-asserts eval on its frozen submodules on every "
                "forward, and this split does not reproduce that call. "
                "Call model.eval() first."
            )

        select_layer = self._backbone.select_layer
        n_layers = len(self._eagle.language_model.model.layers)
        if select_layer != n_layers:
            raise RuntimeError(
                f"backbone.select_layer={select_layer} but the language model has "
                f"{n_layers} layers. hidden_states[select_layer] is only the "
                "final-normed output when the two agree; otherwise it is a "
                "pre-layer input that never passed the final norm."
            )

        if verify_upstream:
            self._verify_upstream_forward()
            self._verify_upstream_action_head()

        self._timer = timer if timer is not None else SystemTimer(enabled=False)
        device_type = self._infer_device_type()
        self._device_type = device_type
        probe_backend = "cuda" if device_type == "cuda" else "cpu"
        # ``stage2_action`` keeps its historical name for the full action-head
        # run so existing CSV consumers read unchanged; ``stage3_warm`` matches
        # the Pi0.5 interceptor's probe for the resumed loop.
        for probe in ("stage1_vision", "stage2_llm", "stage2_action", "stage3_warm"):
            self._timer.register_probe(probe, backend=probe_backend)

        # Optional compiled vision tower (owner directive 2026-08-22): the
        # stage1 forward is launch-bound at B=1, so mode="reduce-overhead"
        # (CUDA graphs) recovers most of the launch overhead. Compilation is
        # cached persistently when TORCHINDUCTOR_CACHE_DIR is set (the serving
        # entrypoint does this), so later server starts skip the compile. The
        # compiled callable is PROCESS-shared via a module registry: concurrent
        # serving builds one runner per connection, and per-runner compiles
        # would each capture their own CUDA graph pool. The FIRST real call
        # double-runs eager vs compiled and refuses to serve on divergence —
        # keys built from a miscompiled tower would be quietly wrong
        # everywhere downstream.
        self._compiled_entry = None
        if compile_vision:
            entry = _COMPILED_VISION_REGISTRY.get(id(self._eagle))
            if entry is None:
                # Mode knob for divergence triage (real 4090, 2026-08-22:
                # reduce-overhead failed the equivalence gate at cos 0.87).
                mode = os.environ.get("OPENPI_STAGE1_COMPILE_MODE", "reduce-overhead")
                entry = {
                    "fn": torch.compile(
                        self._eagle.extract_feature,
                        mode=None if mode in ("", "default") else mode,
                        dynamic=False,
                    ),
                    "checked": False,
                }
                _COMPILED_VISION_REGISTRY[id(self._eagle)] = entry
            self._compiled_entry = entry

    # -- upstream drift guard -------------------------------------------

    def _verify_upstream_forward(self) -> None:
        forward = type(self._eagle).forward
        try:
            source = inspect.getsource(forward)
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                "Cannot read the source of "
                f"{type(self._eagle).__module__}.{type(self._eagle).__name__}.forward "
                "to verify it against the pinned hash. Pass verify_upstream=False "
                "only if you have checked the split by other means."
            ) from exc
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != UPSTREAM_FORWARD_SHA256:
            raise RuntimeError(
                "Upstream forward has changed; the copied stage-1 statements may "
                "no longer match it.\n"
                f"  module:   {type(self._eagle).__module__}\n"
                f"  expected: {UPSTREAM_FORWARD_SHA256}\n"
                f"  actual:   {digest}\n"
                "Re-read the upstream forward, update run_stage1 if the "
                "vision-scatter block changed, then repin the hash and re-run "
                "the two-stage equivalence gate."
            )

    def _verify_upstream_action_head(self) -> None:
        head = self._model.action_head
        try:
            path = inspect.getsourcefile(type(head))
            data = open(path, "rb").read() if path else None  # noqa: SIM115
        except (OSError, TypeError) as exc:
            raise RuntimeError(
                "Cannot read the source file of "
                f"{type(head).__module__}.{type(head).__name__} to verify it "
                "against the pinned hash. Pass verify_upstream=False only if "
                "you have checked the stage-3 transcription by other means."
            ) from exc
        if data is None:
            raise RuntimeError(
                f"{type(head).__name__} has no source file; cannot pin the action head."
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != UPSTREAM_ACTION_HEAD_SHA256:
            raise RuntimeError(
                "Upstream action head has changed; denoise_loop may no longer "
                "transcribe its get_action.\n"
                f"  file:     {path}\n"
                f"  expected: {UPSTREAM_ACTION_HEAD_SHA256}\n"
                f"  actual:   {digest}\n"
                "Re-read get_action, update denoise_loop / denoise_step, repin "
                "the hash and re-run the three-stage equivalence gate."
            )

    def live_schedule(self) -> DenoiseSchedule:
        """The schedule the action head runs *right now*, from its live step count."""
        return groot_n15_schedule(self._model.action_head.num_inference_timesteps)

    def _infer_device_type(self) -> str:
        device = getattr(self._model, "device", None)
        if device is not None:
            return torch.device(device).type
        for param in getattr(self._model, "parameters", lambda: iter(()))():
            return param.device.type
        return "cpu"

    # -- execution context ----------------------------------------------

    @contextlib.contextmanager
    def session(self) -> Iterator[None]:
        """Inference + autocast context both stages must run inside.

        Mirrors what ``Gr00tPolicy`` wraps its own single-shot call in, so the
        split reproduces the unsplit numbers. Entering it twice — once per
        stage, with a cache lookup in between — is safe: autocast's casts are
        deterministic and idempotent, and this model's parameters are already
        bf16 so there is no fp32 weight cache to invalidate.
        """
        with torch.inference_mode(), torch.autocast(self._device_type, torch.bfloat16):
            yield

    def _require_session(self, stage: str) -> None:
        dtype = _autocast_dtype(self._device_type)
        if dtype is not torch.bfloat16:
            raise RuntimeError(
                f"{stage} must run inside GrootStagedRunner.session(): autocast "
                f"is {'off' if dtype is None else dtype} but bfloat16 is required. "
                "Outside the context LayerNorm stays in bf16 instead of being "
                "promoted to fp32, so this call would produce keys and actions "
                "that silently disagree with every other code path."
            )

    # -- stages ----------------------------------------------------------

    def run_stage1(self, normalized_input: dict) -> GrootStage1Output:
        """Vision tower + language embedding + scatter, stopping at the LLM's door."""
        self._require_session("run_stage1")

        backbone_inputs, action_inputs = self._model.prepare_input(normalized_input)
        eagle_input = {
            k.removeprefix("eagle_"): v
            for k, v in backbone_inputs.items()
            if k.startswith("eagle_")
        }
        eagle_input.pop("image_sizes", None)
        if set(eagle_input) != _EAGLE_INPUT_KEYS:
            raise RuntimeError(
                f"Unexpected eagle inputs {sorted(eagle_input)}; this split copies "
                f"the upstream forward for exactly {sorted(_EAGLE_INPUT_KEYS)}. "
                "A new key means upstream would consume something this copy drops."
            )

        input_ids = eagle_input["input_ids"]
        if input_ids.shape[0] != 1:
            raise RuntimeError(
                f"batch size {input_ids.shape[0]}; only B=1 is supported. The "
                "eagle tokenizer pads on the left, so a batch would shift the "
                "image-token runs per row and invalidate the key layout."
            )

        with self._timer.measure("stage1_vision"):
            # --- copied from upstream forward, lines 235-259 ---
            input_embeds = self._eagle.language_model.get_input_embeddings()(input_ids)
            if self._compiled_entry is None:
                vit_embeds = self._eagle.extract_feature(eagle_input["pixel_values"])
            else:
                # Clone unconditionally, not just on the first (checked) call.
                # Under mode="reduce-overhead" the compiled callable returns a
                # CUDA-graph static output buffer, and the registry entry is
                # shared across every runner built on the same base model -- so a
                # second stage1 on another connection overwrites the tensor this
                # one is still scattering into the language sequence, and every
                # retrieval key downstream inherits the wrong image tokens with
                # nothing raising. The lock around inference is what prevents
                # that today; this makes the output side safe on its own terms.
                # It is not sufficient on its own: the graph's *input* buffer is
                # static too, so a concurrent caller can still replay on another
                # caller's pixels. The lock remains required.
                vit_embeds = self._compiled_entry["fn"](
                    eagle_input["pixel_values"]
                ).clone()
                if not self._compiled_entry["checked"]:
                    vit_embeds = self._verify_compiled_vision(
                        eagle_input["pixel_values"],
                        vit_embeds,
                    )

            b, n, c = input_embeds.shape
            flat_embeds = input_embeds.reshape(b * n, c)
            flat_ids = input_ids.reshape(b * n)
            selected = flat_ids == self._eagle.image_token_index

            n_image_tokens = int(selected.sum())
            n_expected = vit_embeds.shape[0] * vit_embeds.shape[1]
            if n_image_tokens != n_expected:
                # Upstream reshapes and truncates here. We refuse instead: a
                # truncated scatter puts image content at the wrong positions,
                # and every key built afterwards would be quietly wrong.
                raise RuntimeError(
                    f"{n_image_tokens} image tokens in the prompt but the vision "
                    f"tower produced {n_expected} embeddings."
                )
            flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, c
            )
            input_embeds = flat_embeds.reshape(b, n, c)
            # --- end copied block ---

        return GrootStage1Output(
            input_embeds=input_embeds,
            attention_mask=eagle_input["attention_mask"],
            image_token_mask=selected.reshape(b, n),
            action_inputs=action_inputs,
        )

    def _verify_compiled_vision(
        self,
        pixel_values: torch.Tensor,
        compiled_out: torch.Tensor,
    ) -> torch.Tensor:
        """One-time eager-vs-compiled gate on the FIRST real input.

        Compiled kernels reorder bf16 reductions, so bitwise equality is not
        the bar; per-token cosine against the eager tower is. On divergence we
        raise instead of serving: every retrieval key downstream inherits this
        tensor.

        ``compiled_out`` is already a clone -- ``run_stage1`` copies it out of
        the CUDA-graph static buffer before calling -- so the eager re-run below
        cannot alias it and no second copy is taken here.
        """
        eager_out = self._eagle.extract_feature(pixel_values)
        a = compiled_out.float().reshape(-1, compiled_out.shape[-1])
        b = eager_out.float().reshape(-1, eager_out.shape[-1])
        cos = F.cosine_similarity(a, b, dim=-1)
        worst = float(cos.min())
        logger.info("compiled vision tower one-time check: worst token cos=%.6f", worst)
        if worst < 0.999:
            raise RuntimeError(
                f"compiled vision tower diverges from eager (worst token cosine "
                f"{worst:.6f} < 0.999); refusing to serve miscompiled keys."
            )
        self._compiled_entry["checked"] = True
        return compiled_out

    def run_stage2(self, stage1: GrootStage1Output) -> GrootStage2Output:
        """Language model + full action head: the MISS path, upstream's own loop."""
        stage2 = self.run_stage2_llm(stage1)
        stage3 = self.run_stage3(stage2)
        return GrootStage2Output(
            backbone_features=stage2.backbone_features,
            attention_mask=stage2.attention_mask,
            action_inputs=stage2.action_inputs,
            action_pred=stage3.action_pred,
        )

    def run_stage2_llm(self, stage1: GrootStage1Output) -> GrootStage2Output:
        """Language model only, stopping at the backbone features the head consumes."""
        self._require_session("run_stage2_llm")

        with self._timer.measure("stage2_llm"):
            # Argument list mirrors upstream's call verbatim. `return_dict` is
            # deliberately absent: upstream does not forward it either, and the
            # language model defaults to returning an object.
            outputs = self._eagle.language_model(
                inputs_embeds=stage1.input_embeds,
                attention_mask=stage1.attention_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=True,
            )
            features = outputs.hidden_states[self._backbone.select_layer]
            features = self._backbone.eagle_linear(features)

        return GrootStage2Output(
            backbone_features=features,
            attention_mask=stage1.attention_mask,
            action_inputs=stage1.action_inputs,
        )

    def _head_inputs(self, stage2: GrootStage2Output) -> Any:
        # Rebuilt every call, never cached on the stage-2 output: the action
        # head's process_backbone_output writes the normalised features back
        # into this mapping in place, so a reused object would get its
        # LayerNorm applied twice.
        return _batch_feature(
            {
                "backbone_features": stage2.backbone_features,
                "backbone_attention_mask": stage2.attention_mask,
            }
        )

    def run_stage3(
        self, stage2: GrootStage2Output, *, noise: Optional[torch.Tensor] = None
    ) -> GrootStage3Output:
        """Full flow-matching loop from pure noise.

        With ``noise=None`` this is upstream's ``get_action`` verbatim -- the
        production MISS path. Passing ``noise`` runs the pinned transcription
        instead, which is what the equivalence gate and the benchmark use to
        compare the two under identical inputs.
        """
        self._require_session("run_stage3")
        backbone_outputs = self._head_inputs(stage2)
        head = self._model.action_head
        with self._timer.measure("stage2_action"):
            if noise is None:
                action_head_outputs = head.get_action(
                    backbone_outputs, stage2.action_inputs
                )
                action_pred = action_head_outputs["action_pred"]
            else:
                action_pred = denoise_loop(
                    head,
                    backbone_outputs,
                    stage2.action_inputs,
                    noise=noise,
                    num_steps=head.num_inference_timesteps,
                )
                action_head_outputs = _batch_feature({"action_pred": action_pred})
        self._model.validate_data(
            action_head_outputs, backbone_outputs, is_training=False
        )
        return GrootStage3Output(
            action_pred=action_pred,
            start_t=None,
            steps_run=head.num_inference_timesteps,
        )

    def run_stage3_from(
        self,
        stage2: GrootStage2Output,
        start_x: torch.Tensor,
        start_t: float,
        *,
        schedule: DenoiseSchedule,
    ) -> GrootStage3Output:
        """Resume the flow-matching loop from a cached snapshot: the WARM_START path.

        ``schedule`` is the library's stamp; it must be the schedule the head is
        running right now, otherwise ``start_t`` names a different step than
        the one ``start_x`` was taken from. Step arithmetic goes through the
        schedule object -- ``remaining_steps`` is ``N - i`` for this ascending
        loop, which is the *opposite* of what the Pi0.5 formula would give.
        """
        self._require_session("run_stage3_from")
        live = self.live_schedule()
        if schedule != live:
            raise RuntimeError(
                f"run_stage3_from: library schedule {schedule.schedule_id} but the "
                f"action head is running {live.schedule_id}; a snapshot at "
                f"t={start_t} would be resumed at the wrong step."
            )
        start_index = schedule.snapshot_index(start_t)
        backbone_outputs = self._head_inputs(stage2)
        head = self._model.action_head
        if start_x.dim() == 2:
            start_x = start_x[None, ...]
        with self._timer.measure("stage3_warm"):
            action_pred = denoise_loop(
                head,
                backbone_outputs,
                stage2.action_inputs,
                noise=start_x,
                num_steps=schedule.num_steps,
                start_index=start_index,
            )
        self._model.validate_data(
            _batch_feature({"action_pred": action_pred}),
            backbone_outputs,
            is_training=False,
        )
        return GrootStage3Output(
            action_pred=action_pred,
            start_t=start_t,
            steps_run=schedule.remaining_steps(start_t),
        )


# ------------------------------------------------------------------
# Transcription of upstream FlowmatchingActionHead.get_action
# ------------------------------------------------------------------


def denoise_step(
    action_head: Any,
    vl: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    actions: torch.Tensor,
    timesteps_tensor: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One Euler step of the upstream loop: encode, DiT, decode, ``x + dt * v``."""
    action_features = action_head.action_encoder(
        actions, timesteps_tensor, embodiment_id
    )
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(
            action_features.shape[1], dtype=torch.long, device=vl.device
        )
        action_features = action_features + action_head.position_embedding(
            pos_ids
        ).unsqueeze(0)
    future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
        vl.shape[0], -1, -1
    )
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
    model_output = action_head.model(
        hidden_states=sa_embs, encoder_hidden_states=vl, timestep=timesteps_tensor
    )
    pred = action_head.action_decoder(model_output, embodiment_id)
    return actions + dt * pred[:, -action_head.action_horizon :]


def denoise_loop(
    action_head: Any,
    backbone_output: Any,
    action_input: Any,
    *,
    noise: torch.Tensor,
    num_steps: int,
    start_index: int = 0,
    step_fn: Callable[..., torch.Tensor] = denoise_step,
) -> torch.Tensor:
    """Upstream ``get_action`` with the noise hoisted out and a resume point.

    Upstream draws ``torch.randn`` inside the function body; here ``noise`` is
    the loop's starting chunk -- pure noise for a full run, or a cached
    snapshot when ``start_index`` is the step that snapshot feeds. Everything
    else is upstream's, character for character: the **ascending**
    ``t_cont = t/N``, the integer bucket discretisation, the position
    embedding, the ``future_tokens`` expansion and the ``+dt*v`` Euler update.
    ``step_fn`` exists so a caller may substitute a compiled single step.
    ``action_input`` is indexed rather than attribute-accessed so the loop also
    runs against plain-dict stand-ins; ``BatchFeature`` supports both.
    """
    processed = action_head.process_backbone_output(backbone_output)
    vl = processed.backbone_features
    embodiment_id = action_input["embodiment_id"]
    state_features = action_head.state_encoder(action_input["state"], embodiment_id)

    batch_size = vl.shape[0]
    actions = noise.to(device=vl.device, dtype=vl.dtype)
    dt = 1.0 / num_steps
    for t in range(start_index, num_steps):
        t_cont = t / float(num_steps)  # ascending: 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=vl.device
        )
        # A reduce-overhead graph returns a static output buffer. Clone after
        # every step because the next denoise step consumes this value.
        actions = step_fn(
            action_head,
            vl,
            state_features,
            embodiment_id,
            actions,
            timesteps_tensor,
            dt,
        ).clone()
    return actions


def _batch_feature(data: dict) -> Any:
    """Wrap in transformers' BatchFeature, falling back to a dict for stubs."""
    try:
        from transformers.feature_extraction_utils import BatchFeature
    except ImportError:  # pragma: no cover - transformers is always present in prod
        return data
    return BatchFeature(data=data)
