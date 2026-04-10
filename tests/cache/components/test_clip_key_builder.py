"""Tests for CLIPKeyBuilder and shared helpers."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Install a lightweight fake open_clip module BEFORE importing the builder,
# so _ensure_model_loaded() never triggers a real open_clip import.
# ---------------------------------------------------------------------------

MOCK_EMBED_DIM = 16


def _make_mock_model():
    """Create a fake CLIP model with encode_image side-effect."""
    mock_model = MagicMock()
    mock_model.visual.output_dim = MOCK_EMBED_DIM
    mock_model.eval = MagicMock()

    def _encode_image(batch, normalize=False):
        n = batch.shape[0]
        embs = torch.randn(n, MOCK_EMBED_DIM)
        if normalize:
            embs = torch.nn.functional.normalize(embs, dim=-1)
        return embs

    mock_model.encode_image = MagicMock(side_effect=_encode_image)
    return mock_model


def _preprocess(pil_img):
    """Simplified preprocess: PIL -> [3, H, W] float tensor."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


_FAKE_MODEL = _make_mock_model()

_fake_open_clip = MagicMock()
_fake_open_clip.create_model_and_transforms = MagicMock(
    return_value=(_FAKE_MODEL, None, _preprocess),
)
sys.modules.setdefault("open_clip", _fake_open_clip)

# Now safe to import — open_clip is already in sys.modules.
from openpi.cache.components.clip_key_builder import (  # noqa: E402
    CLIPKeyBuilder,
    clip_prompt_key_from_tokens,
    clip_state_key,
)
from openpi.cache.types import (  # noqa: E402
    PROMPT_EMB,
    ROBOT_STATE,
    VISION_0,
    VISION_1,
    VISION_2,
    CheckpointID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage1(prefix_len: int = 800, emb_dim: int = 2048, state_dim: int = 32):
    """Create a fake Stage1Output with realistic shapes."""
    return SimpleNamespace(
        prefix_embs=torch.randn(1, prefix_len, emb_dim),
        state=torch.randn(1, state_dim),
    )


def _make_images(slots=("base_0_rgb", "left_wrist_0_rgb")):
    """Create fake input_images dict with 224x224 uint8 images."""
    return {slot: np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for slot in slots}


def _fresh_builder(enabled_fields=None):
    """Create a CLIPKeyBuilder with fresh mock model and force load."""
    model = _make_mock_model()
    _fake_open_clip.create_model_and_transforms.return_value = (model, None, _preprocess)
    builder = CLIPKeyBuilder(
        clip_model_name="ViT-B-32",
        clip_pretrained="openai",
        enabled_fields=enabled_fields,
    )
    builder._ensure_model_loaded(torch.device("cpu"))
    return builder


@pytest.fixture()
def clip_builder():
    """Create a CLIPKeyBuilder with mocked CLIP model."""
    return _fresh_builder(enabled_fields=[VISION_0, VISION_1, ROBOT_STATE])


# ---------------------------------------------------------------------------
# Shared helper tests
# ---------------------------------------------------------------------------


def test_clip_prompt_key_from_tokens_shape():
    tokens = torch.randn(50, 2048)
    key = clip_prompt_key_from_tokens(tokens)
    assert key.shape == (2048,)
    assert key.dtype == torch.float32
    assert key.is_contiguous()


def test_clip_prompt_key_from_tokens_is_mean():
    tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    key = clip_prompt_key_from_tokens(tokens)
    expected = torch.tensor([2.0, 3.0])
    assert torch.allclose(key, expected)


def test_clip_state_key_shape():
    state = torch.randn(32)
    key = clip_state_key(state)
    assert key.shape == (32,)
    assert key.dtype == torch.float32
    assert key.is_contiguous()


# ---------------------------------------------------------------------------
# CLIPKeyBuilder: collect / build
# ---------------------------------------------------------------------------


def test_collect_stores_stage1_tensors(clip_builder):
    stage1 = _make_stage1()
    images = _make_images()
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)

    assert "prefix_embs" in clip_builder.cached_data
    assert "state" in clip_builder.cached_data


def test_collect_does_not_expose_images(clip_builder):
    """cached_data must only contain tensors, not numpy images."""
    stage1 = _make_stage1()
    images = _make_images()
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)

    for v in clip_builder.cached_data.values():
        assert isinstance(v, torch.Tensor)


def test_build_returns_enabled_vision_and_state(clip_builder):
    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb", "left_wrist_0_rgb"))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = clip_builder.build(CheckpointID.CP1)

    assert VISION_0 in keys
    assert VISION_1 in keys
    assert ROBOT_STATE in keys
    assert VISION_2 not in keys
    assert PROMPT_EMB not in keys


def test_build_vision_dim(clip_builder):
    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb",))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = clip_builder.build(CheckpointID.CP1)

    assert keys[VISION_0].shape == (MOCK_EMBED_DIM,)
    assert keys[VISION_0].dtype == torch.float32


def test_build_vision_is_normalized(clip_builder):
    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb",))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = clip_builder.build(CheckpointID.CP1)

    norm = torch.norm(keys[VISION_0]).item()
    assert abs(norm - 1.0) < 0.01


def test_build_missing_image_skips_field(clip_builder):
    """If an enabled vision field's image is missing, skip it (no zero vector)."""
    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb",))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = clip_builder.build(CheckpointID.CP1)

    assert VISION_0 in keys
    assert VISION_1 not in keys


def test_build_no_images_still_returns_state(clip_builder):
    """Even without images, robot_state should still be produced."""
    stage1 = _make_stage1()
    clip_builder.collect(CheckpointID.CP1, stage1=stage1)
    keys = clip_builder.build(CheckpointID.CP1)

    assert ROBOT_STATE in keys
    assert VISION_0 not in keys


def test_build_with_prompt_emb_enabled():
    """When prompt_emb is enabled, it should be produced from prefix_embs."""
    builder = _fresh_builder(enabled_fields=[VISION_0, PROMPT_EMB, ROBOT_STATE])

    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb",))
    builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = builder.build(CheckpointID.CP1)

    assert PROMPT_EMB in keys
    assert keys[PROMPT_EMB].shape == (2048,)


def test_build_batch_encoding(clip_builder):
    """Multiple enabled vision images should be batched in one forward pass."""
    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb", "left_wrist_0_rgb"))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = clip_builder.build(CheckpointID.CP1)

    clip_builder._clip_model.encode_image.assert_called_once()
    call_args = clip_builder._clip_model.encode_image.call_args
    batch_tensor = call_args[0][0]
    assert batch_tensor.shape[0] == 2


def test_clear_resets_state(clip_builder):
    stage1 = _make_stage1()
    images = _make_images()
    clip_builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    clip_builder.clear()

    assert len(clip_builder.cached_data) == 0
    assert clip_builder._images is None


def test_second_collect_clears_previous(clip_builder):
    stage1_a = _make_stage1()
    images_a = _make_images(slots=("base_0_rgb",))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1_a, input_images=images_a)

    stage1_b = _make_stage1()
    images_b = _make_images(slots=("base_0_rgb", "left_wrist_0_rgb"))
    clip_builder.collect(CheckpointID.CP1, stage1=stage1_b, input_images=images_b)

    assert len(clip_builder._images) == 2


def test_unsupported_checkpoint_raises(clip_builder):
    stage1 = _make_stage1()
    clip_builder.collect(CheckpointID.CP1, stage1=stage1)
    with pytest.raises(ValueError, match="Unsupported"):
        clip_builder.build(CheckpointID.CP2)


def test_enabled_fields_none_enables_all():
    """When enabled_fields=None, all fields should be produced."""
    builder = _fresh_builder(enabled_fields=None)

    stage1 = _make_stage1()
    images = _make_images(slots=("base_0_rgb", "left_wrist_0_rgb"))
    builder.collect(CheckpointID.CP1, stage1=stage1, input_images=images)
    keys = builder.build(CheckpointID.CP1)

    assert VISION_0 in keys
    assert VISION_1 in keys
    assert PROMPT_EMB in keys
    assert ROBOT_STATE in keys
