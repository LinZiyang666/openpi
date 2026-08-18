"""Cache key builders for GR00T N1.5, cutting keys out of the stage-1 sequence.

Why these cannot reuse Pi0.5's slicing
--------------------------------------
Pi0.5's prefix is three 256-token image blocks followed by the prompt, always
in that order and always at the same offsets, so its builders slice at fixed
positions. GR00T's sequence is produced by a chat template that places the
INSTRUCTION AFTER the image blocks, so the runs start at 20, 283 and 546 and
stay there regardless of prompt length (measured 812/814/834 total tokens for
three prompts, runs pinned in all of them). What breaks a copied Pi0.5 table
is not offset motion but the offsets themselves: 20/283/546 != 0/256/512, so
the fixed table would slice into system/text tokens and still return
correctly-shaped vectors whose distances mean nothing. The runs are therefore
located per step from the image-token mask — correct for this template and
robust to any future template change.

Everything after the slice — the pooling, the CPU transfer, the field
vocabulary — is shared with Pi0.5, so the resulting keys live in the same
space and the same YAML recipes describe both.

One deliberate difference: the slices are cast to float32 *before* pooling.
Online, stage 1 produces bf16 under autocast; offline, the artifact builder
reads fp16 from HDF5 and calls `.float()` before pooling. Pooling in different
dtypes would make the two paths disagree by construction, so the online side
matches the offline one.

Coupling map:
  DEPENDS ON:  GrootStage1Output, Pi0.5's pooling helpers + field constants
  CONSUMED BY: CacheOrchestrator (via config.py's key_builder factory)
  IF CHANGED:  the online/offline key parity gate must be re-run
"""

from __future__ import annotations

import torch

from openpi.cache.components.key_builder import (
    _CP1BaseKeyBuilder,
    _max_pool_tokens,
    _mean_pool_tokens,
    _spatial_pool_tokens,
)
from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
)

# Camera order follows the checkpoint's video-key order, which is what the
# transform chain feeds the chat template: agentview_left, agentview_right,
# eye_in_hand.
_VISION_FIELDS = (VISION_0, VISION_1, VISION_2)
_TOKENS_PER_IMAGE = 256


# ------------------------------------------------------------------
# Mask-driven slicing
# ------------------------------------------------------------------


def _contiguous_runs(mask_row: torch.Tensor) -> list[tuple[int, int]]:
    """Return (start, length) for each maximal run of True in a 1-D bool tensor."""
    idx = torch.nonzero(mask_row, as_tuple=False).flatten().tolist()
    if not idx:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev - start + 1))
            start = i
        prev = i
    runs.append((start, prev - start + 1))
    return runs


def slice_groot_cp1_fields(
    input_embeds: torch.Tensor,
    image_token_mask: torch.Tensor,
    state: torch.Tensor,
    state_mask: torch.Tensor,
    enabled: set[str] | None,
    *,
    expected_state_index: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Cut per-modality token sequences out of one stage-1 sequence.

    Returns float32 tensors on the input device: `vision_*` are
    ``[256, emb_dim]``, `prompt_emb` is ``[num_text_tokens, emb_dim]`` and
    `robot_state` is the valid slice of the normalised state vector.

    Raises RuntimeError when the image-token layout is not the expected three
    equal runs. That is a hard failure on purpose: every downstream number
    stays plausible when the slicing is wrong, so this is the only place the
    mistake is still visible.
    """
    if input_embeds.shape[0] != 1:
        raise RuntimeError(f"expected batch size 1, got {input_embeds.shape[0]}")

    runs = _contiguous_runs(image_token_mask[0])
    if len(runs) != len(_VISION_FIELDS):
        raise RuntimeError(
            f"expected {len(_VISION_FIELDS)} contiguous image-token runs, found "
            f"{len(runs)}: {runs}. The camera-to-field mapping is defined by run "
            "order, so any other count means the prompt layout changed."
        )
    bad = [(start, length) for start, length in runs if length != _TOKENS_PER_IMAGE]
    if bad:
        raise RuntimeError(
            f"image-token runs must be {_TOKENS_PER_IMAGE} tokens each, got {runs}."
        )

    seq = input_embeds[0]
    result: dict[str, torch.Tensor] = {}

    for field_name, (start, length) in zip(_VISION_FIELDS, runs, strict=True):
        if enabled is not None and field_name not in enabled:
            continue
        result[field_name] = seq[start : start + length].float()

    if enabled is None or PROMPT_EMB in enabled:
        # Everything the vision embeddings did not overwrite. Length varies
        # with the instruction; the reducers pool over the token axis so that
        # is fine, but it does mean prompt keys are constant within an episode.
        result[PROMPT_EMB] = seq[~image_token_mask[0]].float()

    if enabled is None or ROBOT_STATE in enabled:
        valid = state_mask[0, -1]
        if expected_state_index is not None and not torch.equal(valid, expected_state_index):
            raise RuntimeError(
                "state_mask changed mid-episode; robot_state keys built before "
                "and after this step would have different meanings per position."
            )
        result[ROBOT_STATE] = state[0, -1][valid].float()

    return result


# ------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------


class _GrootCP1BaseKeyBuilder(_CP1BaseKeyBuilder):
    """Shared GR00T half: collect the stage-1 fields, slice them by mask.

    Reduction, CPU transfer and field filtering are inherited from the Pi0.5
    base unchanged, which is what keeps the two key spaces comparable.
    """

    def __init__(self, enabled_fields: list[str] | None = None) -> None:
        super().__init__(enabled_fields=enabled_fields)
        self._state_index: torch.Tensor | None = None

    def collect(self, checkpoint_id, **stage_outputs) -> None:
        self._cache.clear()
        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._cache["input_embeds"] = s1.input_embeds
            self._cache["image_token_mask"] = s1.image_token_mask
            self._cache["state"] = s1.state
            self._cache["state_mask"] = s1.state_mask
        if "stage3" in stage_outputs:
            self._cache["action_chunk"] = stage_outputs["stage3"].action_chunk

    def _slice(self) -> dict[str, torch.Tensor]:
        raw = slice_groot_cp1_fields(
            self._cache["input_embeds"],
            self._cache["image_token_mask"],
            self._cache["state"],
            self._cache["state_mask"],
            self._enabled,
            expected_state_index=self._state_index,
        )
        if self._state_index is None:
            # Latch the first step's valid-dimension set so a later change is
            # caught rather than silently producing keys of a different width.
            self._state_index = self._cache["state_mask"][0, -1].clone()
        return raw

    def on_episode_start(self, *args, **kwargs) -> None:
        self._state_index = None


class GrootCP1MeanPoolKeyBuilder(_GrootCP1BaseKeyBuilder):
    """vision/prompt mean-pooled to [emb_dim]. Output: vision_*=2048, prompt=2048."""

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class GrootCP1SpatialPool16KeyBuilder(_GrootCP1BaseKeyBuilder):
    """4x4 spatial pool. Output: vision_*=16*2048=32768, prompt=2048 (mean)."""

    _GRID_SIZE = 16
    _POOL_SIZE = 4

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class GrootCP1SpatialPool4KeyBuilder(_GrootCP1BaseKeyBuilder):
    """2x2 spatial pool. Output: vision_*=4*2048=8192, prompt=2048 (mean)."""

    _GRID_SIZE = 16
    _POOL_SIZE = 2

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _spatial_pool_tokens(tokens, self._GRID_SIZE, self._POOL_SIZE)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _mean_pool_tokens(tokens)


class GrootCP1MaxPoolKeyBuilder(_GrootCP1BaseKeyBuilder):
    """vision/prompt per-dim max to [emb_dim]. Output: vision_*=2048, prompt=2048."""

    def _reduce_vision(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)

    def _reduce_prompt(self, tokens: torch.Tensor) -> torch.Tensor:
        return _max_pool_tokens(tokens)
