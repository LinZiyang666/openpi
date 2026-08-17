"""The collector, from the server's own construction entry down to the HDF5 file.

A wrapper that is never instantiated collects nothing, so the test starts at
the flag the operator actually types.
"""

from __future__ import annotations

import types

import h5py
import numpy as np
import pytest
import torch

from exp.robocasa365.groot_cache_collector import GrootCacheCollector
from exp.robocasa365.serve_groot_n15 import _build_served_policy
from openpi.cache.groot.staged import GrootStagedRunner

from tests.cache.groot.conftest import ACTION_DIM, ACTION_HORIZON, StubGrootModel


class _StubPolicy:
    def __init__(self, model: StubGrootModel) -> None:
        self.model = model

    def apply_transforms(self, obs):
        return self.model.build_inputs()

    def unapply_transforms(self, action):
        return {"action.out": action["action"].numpy()}


@pytest.fixture
def pinned_to_the_stub(monkeypatch):
    """Accept the stub's own forward as if it were upstream.

    The server builds its runner with verification on, which is the point --
    so the way to exercise that path is to repin the hash, not to disable the
    guard. test_groot_staged.py covers the guard firing.
    """
    import hashlib
    import inspect

    from openpi.cache.groot import staged

    from tests.cache.groot.conftest import _StubEagle

    digest = hashlib.sha256(inspect.getsource(_StubEagle.forward).encode()).hexdigest()
    monkeypatch.setattr(staged, "UPSTREAM_FORWARD_SHA256", digest)


def _args(**overrides):
    base = {
        "cache_config": None,
        "collect_hdf5": None,
        "diagnostic_seed": None,
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _collector(tmp_path):
    model = StubGrootModel()
    policy = _StubPolicy(model)
    runner = GrootStagedRunner(model, verify_upstream=False)
    return model, GrootCacheCollector(policy, runner, out_dir=str(tmp_path))


def _obs():
    return {"state.x": np.zeros((1, 3), dtype=np.float32)}


def _only_h5(tmp_path):
    files = list(tmp_path.rglob("*.h5"))
    assert len(files) == 1, files
    return files[0]


# ------------------------------------------------------------------
# Server construction
# ------------------------------------------------------------------


def test_no_flags_serves_the_bare_teacher(pinned_to_the_stub):
    model = StubGrootModel()
    policy = _StubPolicy(model)
    served, label = _build_served_policy(policy, _args())
    assert served is policy
    assert "teacher-only" in label


def test_collect_flag_installs_the_collector(pinned_to_the_stub, tmp_path):
    model = StubGrootModel()
    policy = _StubPolicy(model)
    served, label = _build_served_policy(policy, _args(collect_hdf5=str(tmp_path)))
    assert isinstance(served, GrootCacheCollector)
    assert str(tmp_path) in label


def test_cache_and_collect_together_are_refused_by_the_parser():
    with pytest.raises(SystemExit):
        _parse(["--cache-config", "x.yaml", "--collect-hdf5", "/tmp/y"])


def test_diagnostic_seed_with_collection_is_refused():
    with pytest.raises(SystemExit):
        _parse(["--collect-hdf5", "/tmp/y", "--diagnostic-seed", "0"])


def _parse(argv):
    """Run the server's own argument parsing, which is where the refusals live."""
    import sys

    from exp.robocasa365 import serve_groot_n15

    original = sys.argv
    sys.argv = ["serve_groot_n15.py", *argv]
    try:
        # main() parses, then loads a checkpoint; the parser errors fire first.
        serve_groot_n15.main()
    finally:
        sys.argv = original


# ------------------------------------------------------------------
# What lands on disk
# ------------------------------------------------------------------


def test_episode_writes_the_schema_the_offline_builder_reads(tmp_path):
    _, collector = _collector(tmp_path)
    collector.on_episode_start(task="OpenCabinet", episode_id=2)
    collector.get_action(_obs())
    collector.get_action(_obs())
    collector.on_episode_end(success=True)

    with h5py.File(_only_h5(tmp_path), "r") as f:
        steps = sorted(k for k in f if k.startswith("step_"))
        assert len(steps) == 2
        group = f[steps[0]]
        for field in ("vision_0", "vision_1", "vision_2"):
            assert group[field].shape[0] == 256
            assert group[field].dtype == np.float16
        assert group["prompt_emb"].dtype == np.float16
        assert group["robot_state"].dtype == np.float32
        assert group["clean_action"].shape == (ACTION_HORIZON, ACTION_DIM)


def test_task_and_success_attributes_are_written(tmp_path):
    """The builder drops any episode missing either; that loss is silent."""
    _, collector = _collector(tmp_path)
    collector.on_episode_start(task="TurnOnSinkFaucet", episode_id=0)
    collector.get_action(_obs())
    collector.on_episode_end(success=True)

    with h5py.File(_only_h5(tmp_path), "r") as f:
        assert f.attrs["task"] == "TurnOnSinkFaucet"
        assert bool(f.attrs["success"]) is True


def test_failed_episode_is_recorded_as_such(tmp_path):
    _, collector = _collector(tmp_path)
    collector.on_episode_start(task="OpenDrawer", episode_id=1)
    collector.get_action(_obs())
    collector.on_episode_end(success=False)
    with h5py.File(_only_h5(tmp_path), "r") as f:
        assert bool(f.attrs["success"]) is False


def test_collector_returns_a_normal_action_dict(tmp_path):
    _, collector = _collector(tmp_path)
    collector.on_episode_start(task="t", episode_id=0)
    out = collector.get_action(_obs())
    assert "action.out" in out
    assert "__hit_meta__" not in out  # teacher-only: nothing to report


def test_recorded_tensors_are_not_inference_tensors(tmp_path):
    """They outlive the inference context inside the episode buffer."""
    model, collector = _collector(tmp_path)
    collector.on_episode_start(task="t", episode_id=0)
    collector.get_action(_obs())
    buffered = collector._collector._buffer[0]  # noqa: SLF001
    for array in [*buffered.vision_embs, buffered.prompt_emb, buffered.clean_action]:
        assert isinstance(array, np.ndarray)
    assert torch.from_numpy(buffered.clean_action).is_inference() is False


# ------------------------------------------------------------------
# The whole chain: server flag -> collector -> HDF5 -> artifact -> backend
# ------------------------------------------------------------------


def test_collected_episode_builds_a_loadable_groot_artifact(pinned_to_the_stub, tmp_path):
    """Everything downstream of the flag, in one go.

    Each link is individually plausible and jointly broken in the obvious ways:
    a schema the builder cannot read, dims that describe two cameras, an
    artifact with no identity. Running the real builder over a really-collected
    file is the only way that shows up.
    """
    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from exp.common.build_in_memory_cache_artifact import build_artifact

    model = StubGrootModel()
    policy = _StubPolicy(model)
    data_dir = tmp_path / "episodes"
    served, _ = _build_served_policy(policy, _args(collect_hdf5=str(data_dir)))

    served.on_episode_start(task="OpenCabinet", episode_id=0)
    served.get_action(_obs())
    served.get_action(_obs())
    served.on_episode_end(success=True)

    artifact = build_artifact(
        str(data_dir),
        "cp1_groot_mean_pool",
        "CP1",
        workers=-1,
        robot_state_dim=5,  # the stub's valid state width
    )

    # Identity: this is what stops a same-dimension library being loaded under
    # the wrong recipe.
    assert artifact["key_builder_type"] == "cp1_groot_mean_pool"
    assert artifact["checkpoint_id"] == "CP1"

    # Geometry: three cameras, not LIBERO's two.
    assert set(artifact["vector_dims"]) == {
        "vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state",
    }
    assert artifact["vector_dims"]["robot_state"] == 5

    # Content: one entry per recorded step, chained.
    assert len(artifact["entries"]) == 2
    first, second = artifact["entries"]
    assert set(first.query_keys) == set(artifact["vector_dims"])
    assert first.payload.action_chunk.shape == (ACTION_HORIZON, ACTION_DIM)
    assert first.payload.task_key == "OpenCabinet"
    assert second.id in first.next_ids

    # And the backend reads its identity back through the sanctioned facade.
    import pickle

    path = tmp_path / "artifact.pkl"
    path.write_bytes(pickle.dumps(artifact))
    backend = InMemoryBackend(vector_dims=artifact["vector_dims"])
    backend.load_artifact(str(path))
    assert backend.artifact_meta == {
        "key_builder_type": "cp1_groot_mean_pool",
        "checkpoint_id": "CP1",
    }


def test_a_failed_episode_yields_no_entries(pinned_to_the_stub, tmp_path):
    """The builder's success filter is silent, so it is worth pinning."""
    from exp.common.build_in_memory_cache_artifact import build_artifact

    model = StubGrootModel()
    policy = _StubPolicy(model)
    data_dir = tmp_path / "episodes"
    served, _ = _build_served_policy(policy, _args(collect_hdf5=str(data_dir)))
    served.on_episode_start(task="OpenDrawer", episode_id=0)
    served.get_action(_obs())
    served.on_episode_end(success=False)

    artifact = build_artifact(
        str(data_dir), "cp1_groot_mean_pool", "CP1", workers=-1, robot_state_dim=5
    )
    assert artifact["entries"] == []
