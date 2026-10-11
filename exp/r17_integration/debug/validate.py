"""P6 v1 serialization and identity checks; no model or simulator imports."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

INTERFACE_VERSION = "r17.p6.v1"
MODEL_KEYS = ("model_id", "checkpoint_identity", "norm_identity", "layout", "feature_spec", "default_sampler")
EPISODE_KEYS = ("campaign", "arm", "bundle_generation", "benchmark_protocol", "task_name",
                "orig_init_idx", "task_data_hash", "task_uid", "attempt")


def plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return plain(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"non-JSON metadata: {type(value).__name__}")


def canonical(value: Any) -> bytes:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_arrays(arrays: dict, *, serialization=None) -> dict[str, np.ndarray]:
    """Copy observer buffers; BF16 stores float32 values plus exact uint16 bits.

    Optional serialization maps original names to their reversible companion
    datasets/dtypes. Torch is imported only for an existing BF16 torch tensor;
    client-only NumPy writers never import a model dependency.
    """
    result = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key or "/" in key:
            raise ValueError("array names must be nonempty flat names")
        bits = None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if str(value.dtype) == "torch.bfloat16":
                import torch
                value = value.contiguous()
                bits = value.view(torch.uint16).numpy().copy(order="C")
                value = value.float().numpy()
            else:
                value = value.numpy()
        a = np.asarray(value)
        if a.dtype.name == "bfloat16":
            bits = np.ascontiguousarray(a).view(np.uint16).copy(order="C")
            a = a.astype(np.float32)
        if bits is not None:
            companion = key + "__r17_bfloat16_bits"
            if companion in arrays:
                raise ValueError("BF16 companion array name collision: " + companion)
            result[companion] = bits
            if serialization is not None:
                serialization[key] = dict(native_dtype="bfloat16", storage_dtype="float32",
                    bits_dtype="uint16", bits_array=companion, original_array=key,
                    encoding="torch_uint16_view" if hasattr(arrays[key], "detach") else "numpy_uint16_view")
        if a.dtype.hasobject or a.dtype.kind not in "biufcSU":
            raise ValueError(f"unsafe array {key}: {a.dtype}")
        result[key] = a.copy(order="C")
    return result


def model_binding(spec: Any, benchmark_protocol: str, *, sampler=None) -> dict:
    spec = plain(spec)
    missing = [k for k in MODEL_KEYS if k not in spec]
    if missing or not benchmark_protocol:
        raise ValueError(f"incomplete model binding: {missing}")
    value = dict(version=INTERFACE_VERSION, model=spec, benchmark_protocol=benchmark_protocol,
                 sampler=plain(sampler) if sampler is not None else spec["default_sampler"])
    # Roundtrip both rejects NaNs and breaks references to caller-owned dicts.
    return json.loads(canonical(value))


def require_binding(actual: dict, expected: dict) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError("model/norm/layout/feature/camera/sampler/protocol binding mismatch")


def episode_identity(record: dict) -> tuple:
    missing = [k for k in EPISODE_KEYS if k not in record]
    if missing:
        raise ValueError(f"incomplete episode identity: {missing}")
    if any(not isinstance(record[k], str) or not record[k] for k in (
            "campaign", "arm", "bundle_generation", "benchmark_protocol", "task_name", "task_data_hash", "task_uid")):
        raise ValueError("episode provenance requires nonempty identity strings")
    if int(record["attempt"]) < 1 or int(record["orig_init_idx"]) < 0:
        raise ValueError("invalid attempt/init index")
    return tuple(record[k] for k in EPISODE_KEYS)


def decision_identity(record: dict) -> tuple:
    if int(record["decision_seq"]) < 0:
        raise ValueError("invalid decision sequence")
    return episode_identity(record) + (int(record["decision_seq"]),)
