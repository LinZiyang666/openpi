"""Explicit external quantile + checkpoint-internal normalization conventions."""
import numpy as np
from PIL import Image
import torch


def quantile_normalize(x, stats):
    lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    return (np.asarray(x, np.float32) - lo) / (hi - lo + 1e-6) * 2 - 1


def quantile_unnormalize(x, stats):
    lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
    # This is openpi Unnormalize's formula.
    return (np.asarray(x, np.float32) + 1) / 2 * (hi - lo + 1e-6) + lo


def image_tensor(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected uint8 HWC RGB")
    rgb = np.asarray(Image.fromarray(image).resize((96, 96), Image.Resampling.BILINEAR)).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255


class LanguageEncoder:
    """Frozen CLIP-L/14 text encoder. No observation or cross-call prediction cache."""
    def __init__(self, path, kind="projected"):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.model = CLIPTextModelWithProjection.from_pretrained(path, local_files_only=True).eval()
        self.model.requires_grad_(False)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if kind not in ("projected", "pooled"):
            raise ValueError("language_kind must be projected or pooled")
        self.kind = kind

    @torch.inference_mode()
    def __call__(self, prompts):
        tokens = self.tokenizer(prompts, add_special_tokens=True, max_length=77,
                                padding=True, truncation=True, return_tensors="pt")
        if self.kind == "pooled":
            return self.model.text_model(**tokens).pooler_output.float()
        return self.model(**tokens).text_embeds.float()
