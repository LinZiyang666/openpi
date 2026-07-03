"""Tests for gate-research collection emission + CheckResult.searched."""

import numpy as np
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.interceptor import InferenceInterceptor
from openpi.cache.orchestrator import CheckResult

_build = InferenceInterceptor._build_collect_meta


def test_check_result_searched_default_true():
    assert CheckResult(hit_type=HitType.MISS).searched is True


def test_build_collect_meta_none_is_placeholder():
    assert _build(None, ("robot_state",)) == {"collect": None, "searched": False}


def test_build_collect_meta_emits_only_requested_fields():
    cr = CheckResult(
        hit_type=HitType.MISS,
        query_keys={
            "robot_state": torch.tensor([1.0, 2.0, 3.0]),
            "vision_0": torch.tensor([0.5, 0.5], dtype=torch.float16),
        },
        searched=True,
    )
    m = _build(cr, ("robot_state",))
    assert m["searched"] is True
    assert set(m["collect"]) == {"robot_state"}  # vision_0 not requested
    arr = m["collect"]["robot_state"]
    assert isinstance(arr, np.ndarray) and arr.dtype == np.float32
    np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])


def test_build_collect_meta_missing_field_degrades_to_none():
    cr = CheckResult(hit_type=HitType.MISS, query_keys={"robot_state": torch.tensor([1.0])})
    m = _build(cr, ("robot_state", "vision_0"))
    assert m["collect"]["vision_0"] is None
    assert m["collect"]["robot_state"] is not None


def test_build_collect_meta_searched_false_propagates():
    cr = CheckResult(
        hit_type=HitType.MISS,
        query_keys={"robot_state": torch.tensor([1.0])},
        searched=False,
    )
    assert _build(cr, ("robot_state",))["searched"] is False


def test_build_collect_meta_empty_query_keys_placeholder():
    cr = CheckResult(hit_type=HitType.MISS, query_keys=None, searched=True)
    assert _build(cr, ("robot_state",)) == {"collect": None, "searched": True}


def test_wire_codec_roundtrip_list_encoding():
    """The client-side ndarray->float32->list codec is msgpack/JSON-safe and lossless."""
    cr = CheckResult(
        hit_type=HitType.MISS,
        query_keys={"robot_state": torch.tensor([1.5, -2.25, 3.0])},
        searched=True,
    )
    arr = _build(cr, ("robot_state",))["collect"]["robot_state"]
    encoded = np.asarray(arr, dtype=np.float32).tolist()
    assert encoded == [1.5, -2.25, 3.0]  # exact (values representable in f32)
    import json

    assert json.loads(json.dumps({"collect": {"robot_state": encoded}}))["collect"]["robot_state"] == encoded


def test_wire_dtype_robot_state_f32_vision_f16():
    """Per-field wire dtype: robot_state float32, vision/prompt float16 (frame bytes)."""
    cr = CheckResult(
        hit_type=HitType.MISS,
        query_keys={
            "robot_state": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "vision_0": torch.tensor([0.5, 0.25], dtype=torch.float32),
            "prompt_emb": torch.tensor([0.1], dtype=torch.float32),
        },
        searched=True,
    )
    m = _build(cr, ("robot_state", "vision_0", "prompt_emb"))
    assert m["collect"]["robot_state"].dtype == np.float32
    assert m["collect"]["vision_0"].dtype == np.float16
    assert m["collect"]["prompt_emb"].dtype == np.float16


def test_end_to_end_ndarray_msgpack_json_parity():
    """Server f16 ndarray -> client upcast f32 -> list survives msgpack + json."""
    import json

    import msgpack
    import msgpack_numpy

    cr = CheckResult(
        hit_type=HitType.MISS,
        query_keys={"vision_0": torch.tensor([0.5, 0.25, -0.75], dtype=torch.float32)},
        searched=True,
    )
    # Server side: f16 ndarray over msgpack_numpy wire.
    wire = msgpack.unpackb(
        msgpack.packb(_build(cr, ("vision_0",)), default=msgpack_numpy.encode),
        object_hook=msgpack_numpy.decode,
        raw=False,
    )
    arr = wire["collect"]["vision_0"]
    assert arr.dtype == np.float16
    # Client codec: upcast f32 -> list, then through plain msgpack + JSON.
    encoded = arr.astype(np.float32).tolist()
    assert msgpack.unpackb(msgpack.packb({"c": encoded}), raw=False)["c"] == encoded
    assert json.loads(json.dumps({"c": encoded}))["c"] == encoded
    # f16 values [0.5, 0.25, -0.75] are exactly representable.
    assert encoded == [0.5, 0.25, -0.75]


def test_codec_preserves_kb_id_through_client_boundary():
    """Server payload (interceptor-injected kb_id) survives the client codec.

    Regression for a real bug: _encode_collect_meta previously dropped kb_id, so
    the episode-summary kb_id was permanently None.
    """
    from examples.libero.collect_util import encode_collect_meta

    cr = CheckResult(
        hit_type=HitType.MISS,
        query_keys={"robot_state": torch.tensor([1.0, 2.0])},
        searched=True,
    )
    server_payload = _build(cr, ("robot_state",))
    server_payload["kb_id"] = "cp1_mean_pool"  # interceptor.infer() injects this
    out = encode_collect_meta(server_payload)
    assert out["kb_id"] == "cp1_mean_pool"  # not dropped
    assert out["collect"]["robot_state"] == [1.0, 2.0]
    assert out["searched"] is True
