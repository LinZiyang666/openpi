"""The trace H5 is a strict superset of the legacy collected H5 (plan §7.2, §12-H).

The same ``InferenceEmbeddings`` written through ``write_step_group`` (legacy)
and through ``H5TraceSink`` must produce identical legacy datasets (names,
dtype, shape, compression, values); the trace file only adds ``trace_``
attrs and the ``step_XXXX/trace`` subgroup. The legacy readers
(``h5_intermediates``) must read both build (noise on) and diagnostic
(noise off) trace files exactly as they read collected files.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from openpi.cache.trace.h5_sink import (
    H5TraceSink,
    TraceWriter,
    read_json_attr,
    write_trace_group,
)
from openpi.cache.trace.types import ATTR_JSON_MAX_BYTES, NOISE_ACTION_RE
from openpi.cache.types import PI05_V1
from openpi.collect.data_collector import write_episode_attrs, write_step_group
from openpi.collect.h5_intermediates import episode_schedule, read_step_intermediates
from tests.cache.trace.conftest import make_identity, make_plan, make_step

LEGACY_KEYS = {
    "vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state", "clean_action",
    "input_images",
}


def _legacy_file(path, steps):
    with h5py.File(path, "w") as f:
        write_episode_attrs(
            f, experiment="exp", task="task", episode_id=1, num_steps=len(steps),
            success=True, episode_attrs={},
        )
        for i, step in enumerate(steps):
            write_step_group(f.create_group(f"step_{i:04d}"), step.legacy)


def _trace_file(out_dir, steps, *, noise: bool):
    w = TraceWriter(str(out_dir), queue_steps=8)
    plan = make_plan(record_noise_actions=noise, save_timesteps=PI05_V1.timesteps if noise else None)
    sink = H5TraceSink(out_dir, plan=plan, writer=w)
    sink.on_episode_start(make_identity(1, "ep"))
    for step in steps:
        sink.begin_step()
        sink.record_step(step)
        sink.finish_step()
    sink.on_episode_end(True)
    assert w.drain(timeout=10).ok
    return out_dir / "exp" / "ep.h5"


def _dataset_signature(ds: h5py.Dataset) -> tuple:
    return (ds.dtype.str, ds.shape, ds.compression, np.asarray(ds).tobytes())


@pytest.mark.parametrize("noise", [False, True])
def test_legacy_keys_are_byte_identical(tmp_path, noise):
    steps = [make_step(i, noise=noise) for i in range(3)]
    legacy = tmp_path / "legacy.h5"
    _legacy_file(legacy, steps)
    trace = _trace_file(tmp_path / "trace", steps, noise=noise)
    with h5py.File(legacy, "r") as fl, h5py.File(trace, "r") as ft:
        assert sorted(fl.keys()) == sorted(ft.keys())
        for name in fl.keys():
            gl, gt = fl[name], ft[name]
            legacy_names = sorted(gl.keys())
            assert sorted(k for k in gt.keys() if k != "trace") == legacy_names
            for key in legacy_names:
                if isinstance(gl[key], h5py.Group):
                    for sub in gl[key].keys():
                        assert _dataset_signature(gl[key][sub]) == _dataset_signature(gt[key][sub])
                else:
                    assert _dataset_signature(gl[key]) == _dataset_signature(gt[key]), key
            assert "trace" in gt
        # Every legacy attr is present with the same value; trace adds only ``trace_`` keys.
        for k in ("experiment_name", "task", "episode_id", "num_steps", "success"):
            assert fl.attrs[k] == ft.attrs[k]
        extra = set(ft.attrs) - set(fl.attrs)
        allowed_prefix = {"trace_"}
        allowed_exact = {"denoise_schedule_id", "denoising_num_steps", "prompt"}
        for k in extra:
            assert any(k.startswith(p) for p in allowed_prefix) or k in allowed_exact, k


@pytest.mark.parametrize("noise", [False, True])
def test_legacy_reader_contract(tmp_path, noise):
    steps = [make_step(i, noise=noise) for i in range(2)]
    trace = _trace_file(tmp_path / "trace", steps, noise=noise)
    with h5py.File(trace, "r") as f:
        schedule = episode_schedule(f)
        assert schedule is not None and schedule.schedule_id == PI05_V1.schedule_id
        inter, n = read_step_intermediates(f["step_0000"], schedule)
        if noise:
            assert n == PI05_V1.num_steps
            assert sorted(inter) == sorted(PI05_V1.timesteps)
            assert "noise_action_0" in f["step_0000"]
        else:
            assert inter is None and n is None
            assert not any(NOISE_ACTION_RE.match(k) for k in f["step_0000"].keys())
        assert bool(f.attrs["trace_noise_actions_recorded"]) is noise


def test_trace_group_never_contains_snapshot_names(tmp_path):
    step = make_step(0)
    step.action_warm = {3: np.zeros((50, 32), np.float32)}
    with h5py.File(tmp_path / "x.h5", "w") as f:
        g = f.create_group("step_0000")
        write_step_group(g, step.legacy)
        write_trace_group(g, step, make_plan())
        names = []
        g["trace"].visit(names.append)
        assert not any(NOISE_ACTION_RE.match(n.rsplit("/", 1)[-1]) for n in names)
        assert "actions/warm_03" in names
        assert "actions/executed" in names and "actions/full_inference" in names


def test_top_level_has_only_step_groups(tmp_path):
    trace = _trace_file(tmp_path / "trace", [make_step(0)], noise=False)
    with h5py.File(trace, "r") as f:
        assert all(k.startswith("step_") for k in f.keys())
        assert list(f.keys()) == ["step_0000"]


def test_sidecar_rows_match_steps(tmp_path):
    steps = [make_step(i) for i in range(4)]
    trace = _trace_file(tmp_path / "trace", steps, noise=False)
    sidecar = trace.with_suffix("").parent / "ep.trace.jsonl"
    lines = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert "_episode" in lines[0]
    assert [row["step"] for row in lines[1:]] == [0, 1, 2, 3]
    assert lines[1]["executed_arm"] == "full_inference"


def test_large_json_attr_spills_to_dataset(tmp_path):
    step = make_step(0)
    step.verdict = {"hit_type": "MISS", "blob": "x" * (ATTR_JSON_MAX_BYTES + 10)}
    with h5py.File(tmp_path / "x.h5", "w") as f:
        g = f.create_group("step_0000")
        write_step_group(g, step.legacy)
        write_trace_group(g, step, make_plan())
        tg = g["trace"]
        assert tg.attrs["verdict_json"] == "@dataset"
        assert "verdict_json" in tg["json"]
        assert read_json_attr(tg, "verdict_json")["blob"].startswith("x")
        assert read_json_attr(tg, "tier_status_json") == {}


def test_raw_image_key_encoding_is_reversible(tmp_path):
    from openpi.cache.trace.h5_sink import decode_dataset_name, encode_dataset_name

    for key in ("observation/image", "a%b", "x__y/z", "plain"):
        assert decode_dataset_name(encode_dataset_name(key)) == key
    step = make_step(0)
    step.raw_images = {"observation/image": np.zeros((2, 2, 3), np.uint8), "a%b": np.ones((2, 2, 3), np.uint8)}
    with h5py.File(tmp_path / "x.h5", "w") as f:
        g = f.create_group("step_0000")
        write_step_group(g, step.legacy)
        write_trace_group(g, step, make_plan())
        rg = g["trace"]["raw_images"]
        mapping = json.loads(rg.attrs["wire_keys_json"])
        assert set(mapping.values()) == {"observation/image", "a%b"}
        for name, wire in mapping.items():
            assert decode_dataset_name(name) == wire
            assert name in rg
