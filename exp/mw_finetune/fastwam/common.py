"""Dependency-light conventions shared by preparation, training, and serving."""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import socket
import subprocess

import numpy as np
from PIL import Image

DATA_REVISION = "b46a1c185a2368660650a82ac5a5083a56722f48"
BASE_REVISION = "53983e1249b4eb4d89ab42b40b8678a882d331ad"
IMAGE_SIZE = 224
HORIZON = 32
VIDEO_OFFSETS = np.arange(0, 33, 4)


def disk_gate(root, required_gb=0, floor_gb=30):
    root = Path(root)
    while not root.exists():
        root = root.parent
    free = shutil.disk_usage(root).free / 1e9
    if free < floor_gb + required_gb:
        raise RuntimeError(f"{root}: {free:.1f} GB free; require {required_gb}+{floor_gb} GB")
    shared = Path("/data/mw_finetune")
    if socket.gethostname() != "weilandserver" and root.resolve().is_relative_to(shared):
        used = int(subprocess.check_output(["du", "-s", "-B1", str(shared)], text=True).split()[0]) / 1e9
        if used + required_gb > 120:
            raise RuntimeError(f"H100 shared budget: {used:.1f}+{required_gb:.1f} GB would exceed120 GB")
    return free


def resize_rgb(image):
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("image must be uint8 HWC RGB")
    return np.asarray(Image.fromarray(a).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))


def validate_observation(obs):
    image = np.asarray(obs["image"])
    state = np.asarray(obs["state"], dtype=np.float32)
    if image.shape != (480, 480, 3) or image.dtype != np.uint8:
        raise ValueError("expected RGB uint8 (480,480,3)")
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("expected finite float32 state (4,)")
    if not isinstance(obs["prompt"], str) or not obs["prompt"].strip():
        raise ValueError("prompt must be nonempty text")
    return image, state, obs["prompt"]


def normalize(value, stats):
    low, high = np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)
    return 2 * (np.asarray(value, dtype=np.float32) - low) / np.maximum(high - low, 1e-6) - 1


def denormalize(value, stats):
    low, high = np.asarray(stats["min"], np.float32), np.asarray(stats["max"], np.float32)
    return (np.asarray(value, dtype=np.float32) + 1) * np.maximum(high - low, 1e-6) / 2 + low


def window_indices(frame, length):
    """Episode-local padding: duplicate the last image, mask padded actions."""
    action = frame + np.arange(HORIZON)
    video = frame + VIDEO_OFFSETS
    return np.minimum(action, length - 1), action >= length, np.minimum(video, length - 1), video >= length


def lr_at(step, total=100_000, warmup=100, peak=1e-4, minimum=2.5e-6):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(max((step - warmup) / max(total - warmup - 1, 1), 0), 1)
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * t)) / 2
