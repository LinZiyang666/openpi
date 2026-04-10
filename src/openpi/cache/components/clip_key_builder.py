"""CLIP-based query key builder for cache lookup.

Uses open_clip to encode model-input images into low-dimensional vectors
as cache keys.  Vision fields are encoded by CLIP; prompt_emb and
robot_state reuse the same logic as existing CP1 builders (slice from
prefix_embs + mean pool / raw).

Data flow:
  Online:  input_images (CPU numpy) + stage1 (GPU) -> collect() -> build() -> query_keys
  Offline: HDF5 input_images + prompt_emb + robot_state -> shared helpers -> query_keys

Coupling map:
  DEPENDS ON:  open_clip (lazy import), types.py field constants,
               key_builder._PROMPT_START (token layout),
               image_extract.MODEL_IMAGE_KEYS (slot name order)
  CONSUMED BY: CacheOrchestrator.check() (online), build_clip_cache_artifact.py (offline)
  FEEDS INTO:  CacheStorage.search() via QuerySpec (must match backend vector_dims)
  IF CHANGED:  Offline artifact script must stay in sync via shared helpers
"""

from __future__ import annotations

import logging
import numpy as np
import torch

from openpi.cache.types import (
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)

logger = logging.getLogger(__name__)

# Image slot name -> cache query field name.
# Order follows image_extract.MODEL_IMAGE_KEYS: (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb).
_SLOT_TO_FIELD: list[tuple[str, str]] = [
    ("base_0_rgb", VISION_0),
    ("left_wrist_0_rgb", VISION_1),
    ("right_wrist_0_rgb", VISION_2),
]

# Token layout constant imported from key_builder.py.
# prompt tokens start at offset 768 in prefix_embs (3 images * 256 tokens each).
_PROMPT_START = 768  # 3 * 256


# ---------------------------------------------------------------------------
# Shared helpers (online builder + offline artifact script)
# ---------------------------------------------------------------------------


def clip_prompt_key_from_tokens(prompt_tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool prompt token sequence to a single vector.

    Args:
        prompt_tokens: [num_tokens, emb_dim] GPU or CPU tensor.

    Returns:
        [emb_dim] CPU float32 contiguous tensor.
    """
    return prompt_tokens.mean(dim=0).cpu().float().contiguous()


def clip_state_key(state: torch.Tensor) -> torch.Tensor:
    """Pass-through robot state as raw vector.

    Args:
        state: [state_dim] GPU or CPU tensor.

    Returns:
        [state_dim] CPU float32 contiguous tensor.
    """
    return state.cpu().float().contiguous()


# ---------------------------------------------------------------------------
# CLIPKeyBuilder
# ---------------------------------------------------------------------------


class CLIPKeyBuilder:
    """CLIP-based cache key builder.

    Vision fields: CLIP image encoder on raw input images -> [embed_dim].
    prompt_emb:    mean pool from prefix_embs prompt segment -> [2048].
    robot_state:   raw state vector -> [state_dim].

    CLIP model is loaded lazily on first collect() call, with device
    inferred from stage1.prefix_embs.device.

    Coupling:
      - DEPENDS ON: open_clip (lazy), Stage1Output fields, image_extract slot names
      - CONSUMED BY: CacheOrchestrator via QueryKeyBuilder protocol
      - cached_data: only GPU tensors (prefix_embs, state); input_images are private
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        enabled_fields: list[str] | None = None,
    ) -> None:
        self._clip_model_name = clip_model_name
        self._clip_pretrained = clip_pretrained
        self._enabled = set(enabled_fields) if enabled_fields is not None else None

        # Lazy-loaded on first collect().
        self._clip_model: torch.nn.Module | None = None
        self._preprocess = None  # torchvision transform
        self._device: torch.device | None = None
        self._embed_dim: int | None = None

        # Per-cycle caches, cleared by clear().
        self._cache: dict[str, torch.Tensor] = {}  # GPU tensors for cached_data
        self._images: dict[str, np.ndarray] | None = None  # private, CPU numpy

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self, device: torch.device) -> None:
        """Load CLIP model on first call, using device from stage1 tensor."""
        if self._clip_model is not None:
            return
        import open_clip

        self._device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._clip_model_name,
            pretrained=self._clip_pretrained,
            device=device,
        )
        model.eval()
        self._clip_model = model
        self._preprocess = preprocess
        self._embed_dim = model.visual.output_dim
        logger.info(
            "CLIPKeyBuilder: loaded %s (pretrained=%s) on %s, embed_dim=%d",
            self._clip_model_name, self._clip_pretrained, device, self._embed_dim,
        )

    # ------------------------------------------------------------------
    # QueryKeyBuilder protocol
    # ------------------------------------------------------------------

    def collect(self, checkpoint_id: CheckpointID, **stage_outputs) -> None:
        """Collect raw data from stage outputs.

        Caches stage1 tensors (GPU) and input_images (CPU numpy).
        Triggers lazy CLIP model loading on first call.
        """
        self._cache.clear()
        self._images = None

        if "stage1" in stage_outputs:
            s1 = stage_outputs["stage1"]
            self._ensure_model_loaded(s1.prefix_embs.device)
            self._cache["prefix_embs"] = s1.prefix_embs  # [B, prefix_len, emb_dim]
            self._cache["state"] = s1.state               # [B, state_dim]

        if "input_images" in stage_outputs:
            self._images = stage_outputs["input_images"]  # {slot: (H,W,3) uint8}

    def build(self, checkpoint_id: CheckpointID) -> dict[str, torch.Tensor]:
        """Build query key vectors from collected data.

        Vision fields: CLIP encode from self._images.
        prompt_emb:    mean pool from prefix_embs prompt segment.
        robot_state:   raw state vector.

        Missing image slots are skipped (no zero vectors).
        Returns {field_name: [dim] CPU float32}.
        """
        if checkpoint_id not in (CheckpointID.CP1, CheckpointID.CP3):
            raise ValueError(f"Unsupported checkpoint_id: {checkpoint_id}")

        keys: dict[str, torch.Tensor] = {}

        # -- Vision fields: CLIP encode --
        keys.update(self._build_vision_keys())

        # -- prompt_emb: mean pool from prefix_embs --
        if self._is_enabled(PROMPT_EMB) and "prefix_embs" in self._cache:
            prefix = self._cache["prefix_embs"][0]  # [prefix_len, emb_dim]
            prompt_tokens = prefix[_PROMPT_START:]   # [num_prompt_tokens, emb_dim]
            keys[PROMPT_EMB] = clip_prompt_key_from_tokens(prompt_tokens)

        # -- robot_state: raw vector --
        if self._is_enabled(ROBOT_STATE) and "state" in self._cache:
            keys[ROBOT_STATE] = clip_state_key(self._cache["state"][0])

        return keys

    @property
    def cached_data(self) -> dict[str, torch.Tensor]:
        """Expose collected GPU tensors for Gate/Judge. No numpy images."""
        return self._cache

    def clear(self) -> None:
        """Release all cached references."""
        self._cache.clear()
        self._images = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_enabled(self, field: str) -> bool:
        return self._enabled is None or field in self._enabled

    def _build_vision_keys(self) -> dict[str, torch.Tensor]:
        """Encode enabled vision slots via CLIP. Skip missing images."""
        if self._images is None or self._clip_model is None:
            return {}

        # Collect (field_name, preprocessed_tensor) pairs for batch encoding.
        batch_fields: list[str] = []
        batch_tensors: list[torch.Tensor] = []

        for slot_name, field_name in _SLOT_TO_FIELD:
            if not self._is_enabled(field_name):
                continue
            if slot_name not in self._images:
                continue  # Missing camera slot -> skip, no zero vector.
            img_np = self._images[slot_name]  # (H, W, 3) uint8
            tensor = self._preprocess_image(img_np)
            batch_fields.append(field_name)
            batch_tensors.append(tensor)

        if not batch_tensors:
            return {}

        # Batch forward through CLIP image encoder.
        batch = torch.stack(batch_tensors).to(self._device)  # [N, 3, H, W]
        with torch.no_grad():
            embeddings = self._clip_model.encode_image(batch, normalize=True)  # [N, embed_dim]

        result: dict[str, torch.Tensor] = {}
        for i, field_name in enumerate(batch_fields):
            result[field_name] = embeddings[i].cpu().float().contiguous()

        return result

    def _preprocess_image(self, img_np: np.ndarray) -> torch.Tensor:
        """Convert uint8 numpy image to preprocessed CLIP input tensor.

        numpy (H, W, 3) uint8 -> PIL Image -> preprocess -> [3, H, W] float tensor.
        """
        from PIL import Image

        pil_img = Image.fromarray(img_np)
        return self._preprocess(pil_img)  # [3, 224, 224]
