"""Per-stage device placement for PI0/PI0.5 models.

Relocates model sub-modules to different devices after checkpoint loading,
enabling memory-efficient configurations like stage1-only GPU inference.

Public interface:
  - ``StageDeviceConfig``: frozen dataclass holding per-stage device strings.
    Construct via ``StageDeviceConfig.create()`` which normalizes and validates.
  - ``relocate_model_stages(model, config)``: move sub-modules in-place and
    re-tie ``lm_head`` ↔ ``embed_tokens`` weight binding.

Dependencies:
  - ``PI0Pytorch`` (model structure and ``pi05`` flag)
  - HuggingFace ``PaliGemmaForConditionalGeneration.tie_weights()``
"""

from __future__ import annotations

import dataclasses
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Module mapping: Stage -> nn.Module attribute paths on PI0Pytorch
# ------------------------------------------------------------------

# Stage 1: vision encoding + prefix embedding (always present)
_STAGE1_MODULES = [
    "paligemma_with_expert.paligemma.model.vision_tower",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens",
]

# Stage 2: LLM backbone (always present)
# NOTE: lm_head is NOT included — it shares weight with embed_tokens (tied).
# After relocation, tie_weights() restores the binding.
_STAGE2_MODULES = [
    "paligemma_with_expert.paligemma.model.language_model.layers",
    "paligemma_with_expert.paligemma.model.language_model.norm",
]

# Stage 3: action expert + projections (always present)
_STAGE3_MODULES_REQUIRED = [
    "paligemma_with_expert.gemma_expert",
    "action_in_proj",
    "action_out_proj",
]

# Stage 3: conditional on model.pi05
_STAGE3_MODULES_PI05 = ["time_mlp_in", "time_mlp_out"]
_STAGE3_MODULES_PI0 = ["state_proj", "action_time_mlp_in", "action_time_mlp_out"]


# ------------------------------------------------------------------
# Device normalization
# ------------------------------------------------------------------

def _normalize_device(device: str) -> str:
    """Normalize device string: ``"cuda"`` -> ``"cuda:0"``, format validation.

    Validates string format only (not runtime GPU availability).

    Raises:
        ValueError: If device string is not recognized or CUDA format is invalid.
    """
    device = device.strip().lower()
    if device in ("cpu", "meta"):
        return device
    if device == "cuda":
        return "cuda:0"
    if device.startswith("cuda:"):
        parts = device.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid CUDA device string: '{device}'. "
                "Expected format 'cuda:N' with exactly one colon."
            )
        try:
            idx = int(parts[1])
        except ValueError:
            raise ValueError(f"Invalid CUDA device index: '{parts[1]}'") from None
        if idx < 0:
            raise ValueError(f"CUDA device index must be non-negative, got {idx}")
        return f"cuda:{idx}"
    raise ValueError(
        f"Unrecognized device: '{device}'. "
        "Expected 'cpu', 'meta', 'cuda', or 'cuda:N'."
    )


# ------------------------------------------------------------------
# StageDeviceConfig
# ------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class StageDeviceConfig:
    """Per-stage device assignment. None = no override (legacy default).

    Construct via classmethod ``create()`` which normalizes and validates.
    Direct construction is allowed for testing but skips validation.
    """

    stage1: str | None = None
    stage2: str | None = None
    stage3: str | None = None

    @classmethod
    def create(
        cls,
        stage1: str | None,
        stage2: str | None,
        stage3: str | None,
    ) -> StageDeviceConfig:
        """Normalize device strings and validate before returning a frozen instance.

        Rules:
          - All three must be None (legacy default) or all three must be non-None.
          - stage1 cannot be ``"meta"``.
          - ``"cuda"`` is normalized to ``"cuda:0"``.
        """
        all_none = stage1 is None and stage2 is None and stage3 is None
        any_none = stage1 is None or stage2 is None or stage3 is None
        if not all_none and any_none:
            raise ValueError(
                "Either set all three stage devices or none. "
                f"Got stage1={stage1}, stage2={stage2}, stage3={stage3}"
            )
        if all_none:
            return cls()
        # Normalize and validate each device string
        stage1 = _normalize_device(stage1)  # type: ignore[arg-type]
        stage2 = _normalize_device(stage2)  # type: ignore[arg-type]
        stage3 = _normalize_device(stage3)  # type: ignore[arg-type]
        if stage1 == "meta":
            raise ValueError(
                "stage1 cannot be 'meta': Stage 1 must always execute "
                "(all inference paths require vision encoding)."
            )
        return cls(stage1=stage1, stage2=stage2, stage3=stage3)

    @property
    def is_legacy_default(self) -> bool:
        """No user override — use existing pytorch_device=None auto-select."""
        return self.stage1 is None and self.stage2 is None and self.stage3 is None

    @property
    def is_all_same_device(self) -> bool:
        """All stages on the same real device — direct load, no post-load relocate."""
        return (
            self.stage1 is not None
            and self.stage1 == self.stage2 == self.stage3
            and self.stage1 != "meta"
        )

    @property
    def needs_relocation(self) -> bool:
        """Different devices or meta — load to CPU then relocate."""
        return not self.is_legacy_default and not self.is_all_same_device

    @property
    def has_meta_stage(self) -> bool:
        """True if any stage uses meta device (stage1 is never meta)."""
        return "meta" in (self.stage2, self.stage3)

    @property
    def primary_device(self) -> str | None:
        """Device for input tensors (stage1's device). None when legacy default."""
        return self.stage1


# ------------------------------------------------------------------
# Model relocation
# ------------------------------------------------------------------

def _resolve_submodule(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated attribute path to a sub-module.

    Raises:
        AttributeError: If any segment of the path does not exist.
    """
    parts = path.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def _move_stage_modules(
    model: nn.Module,
    module_paths: list[str],
    device: str,
    *,
    required: bool = True,
    stage_name: str = "",
) -> int:
    """Move a list of sub-modules to the target device.

    Returns the total number of parameters moved.

    Args:
        model: The root PI0Pytorch model.
        module_paths: Dot-separated attribute paths relative to model.
        device: Target device string.
        required: If True, raise on missing module. If False, skip with warning.
        stage_name: Human-readable stage name for log messages.
    """
    param_count = 0
    for path in module_paths:
        try:
            submodule = _resolve_submodule(model, path)
        except AttributeError:
            if required:
                raise AttributeError(
                    f"Required module '{path}' not found on model. "
                    f"Check module mapping for {stage_name}."
                ) from None
            logger.warning(
                "Optional module '%s' not found on model, skipping (%s).",
                path, stage_name,
            )
            continue
        submodule.to(device)
        param_count += sum(p.numel() for p in submodule.parameters())
    return param_count


def relocate_model_stages(model: nn.Module, config: StageDeviceConfig) -> None:
    """Move model sub-modules to their target devices in-place.

    Must be called AFTER weights are loaded and model is on CPU
    (``Policy.__init__`` calls ``model.to(pytorch_device)``).

    For meta-device stages, parameters become zero-memory meta tensors.
    Required modules that fail to resolve raise an error.
    Optional modules (Pi0/Pi0.5 conditional) are skipped with a warning.

    After all modules are relocated, calls ``paligemma.tie_weights()`` to
    restore ``lm_head.weight`` → ``embed_tokens.weight`` binding (broken
    by submodule ``.to()``).

    Args:
        model: A ``PI0Pytorch`` instance with weights loaded.
        config: Per-stage device configuration (must not be legacy default).
    """
    if config.is_legacy_default:
        return

    pi05 = getattr(model, "pi05", True)

    # Stage 1
    n1 = _move_stage_modules(
        model, _STAGE1_MODULES, config.stage1,
        required=True, stage_name="Stage 1",
    )
    logger.info("Stage 1 -> %s  (%d params)", config.stage1, n1)

    # Stage 2
    n2 = _move_stage_modules(
        model, _STAGE2_MODULES, config.stage2,
        required=True, stage_name="Stage 2",
    )
    logger.info("Stage 2 -> %s  (%d params)", config.stage2, n2)

    # Stage 3: required + conditional
    stage3_modules = list(_STAGE3_MODULES_REQUIRED)
    stage3_optional = _STAGE3_MODULES_PI05 if pi05 else _STAGE3_MODULES_PI0

    n3_req = _move_stage_modules(
        model, stage3_modules, config.stage3,
        required=True, stage_name="Stage 3",
    )
    n3_opt = _move_stage_modules(
        model, stage3_optional, config.stage3,
        required=False, stage_name="Stage 3 (conditional)",
    )
    logger.info("Stage 3 -> %s  (%d params)", config.stage3, n3_req + n3_opt)

    # Re-tie lm_head ↔ embed_tokens (submodule .to() breaks tied weight)
    paligemma = model.paligemma_with_expert.paligemma
    paligemma.tie_weights()

    # Verify tying was restored
    lm_head_w = paligemma.lm_head.weight
    embed_w = paligemma.model.language_model.embed_tokens.weight
    assert lm_head_w is embed_w, (
        "tie_weights() failed to restore lm_head ↔ embed_tokens binding. "
        f"lm_head.device={lm_head_w.device}, embed_tokens.device={embed_w.device}"
    )
    logger.info(
        "Tied weight restored: lm_head.weight is embed_tokens.weight "
        "(device=%s)", embed_w.device,
    )
