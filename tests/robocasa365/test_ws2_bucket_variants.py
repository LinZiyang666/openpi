"""Tests for the library-side bucket map and the ws2 comparison guards.

The bucket map is what turns a ``winner_id`` into an instruction variant, and
the comparison tool is where the two estimands must stay apart; both are
pinned here on synthetic data.
"""

from __future__ import annotations

import json

import pytest
import torch

from exp.robocasa365.build_bucket_variants import bucket_entries, representative_of


class _Entry:
    def __init__(self, entry_id: str, vec: list[float]) -> None:
        self.id = entry_id
        self.query_keys = {"prompt_emb": torch.tensor(vec, dtype=torch.float32)}


def _entries():
    a, b = [1.0, 0.0], [0.0, 1.0]
    return [
        _Entry("OpenDrawer/episode_0000:0", a),
        _Entry("OpenDrawer/episode_0000:1", a),
        _Entry("OpenDrawer/episode_0005:0", a),
        _Entry("CloseFridge/episode_0003:0", b),
    ]


def test_buckets_group_by_exact_prompt_bytes():
    buckets = bucket_entries(_entries())
    assert len(buckets) == 2
    by_task = {tuple(b["tasks"]): b for b in buckets}
    assert by_task[("OpenDrawer",)]["n_entries"] == 3
    assert by_task[("CloseFridge",)]["n_entries"] == 1
    assert all(not b["ambiguous"] for b in buckets)


def test_bucket_indices_follow_ascending_key_order():
    buckets = bucket_entries(_entries())
    assert [b["bucket_index"] for b in buckets] == [0, 1]
    # Deterministic across runs: the same entries produce the same indices.
    assert [b["tasks"] for b in bucket_entries(_entries())] == [b["tasks"] for b in buckets]


def test_cross_task_bucket_is_flagged_ambiguous():
    shared = [0.5, 0.5]
    buckets = bucket_entries([
        _Entry("OpenDrawer/episode_0000:0", shared),
        _Entry("CloseFridge/episode_0001:0", shared),
    ])
    assert len(buckets) == 1
    assert buckets[0]["ambiguous"] is True
    assert buckets[0]["tasks"] == ["CloseFridge", "OpenDrawer"]


def test_entry_without_prompt_emb_is_refused():
    class _Bad:
        id = "x:0"
        query_keys: dict = {}

    with pytest.raises(SystemExit, match="not a text-IVF artifact"):
        bucket_entries([_Bad()])


def test_representative_recovers_the_collection_seed():
    bucket = {"trajectories": ["OpenDrawer/episode_0042", "OpenDrawer/episode_0007"]}
    rep = representative_of(bucket)
    # Smallest relpath wins -> deterministic representative.
    assert rep["trajectory"] == "OpenDrawer/episode_0042"
    assert rep["seed"] == 42
    assert rep["status"] == "skipped"  # until a replay resolves it


def test_representative_parses_the_real_attempt_suffixed_ids():
    """The library's relpath ids carry the collector's ``_aNN`` attempt tail.

    Files are written as ``<Task>/episode_{idx:04d}_a{attempt:02d}.h5``
    (episode_runner episode_name), so an id pattern anchored without the tail
    silently marks every bucket unresolved and empties the whole attribution
    chain.
    """
    rep = representative_of({"trajectories": ["OpenCabinet/episode_0007_a01"]})
    assert rep["seed"] == 7
    assert rep["status"] == "skipped"
    # A retried episode keeps the episode index as its seed.
    assert representative_of({"trajectories": ["OpenCabinet/episode_0013_a03"]})["seed"] == 13


def test_unparsable_trajectory_is_unresolved_not_dropped():
    rep = representative_of({"trajectories": ["OpenDrawer/weird_name"]})
    assert rep["seed"] is None
    assert rep["status"] == "unresolved"


# ------------------------------------------------------------------
# estimand separation in the comparison tool
# ------------------------------------------------------------------


def _cell(success_map):
    return dict(success_map)


def test_decomposition_refuses_cells_without_a_control_pairing(tmp_path, monkeypatch):
    import argparse

    from exp.robocasa365 import analyze_ws2_vs_ws1 as mod

    grid = [("OpenDrawer", 0), ("OpenDrawer", 1)]
    ws1 = {"iso_a": _cell({k: True for k in grid}), "grid_b": _cell({k: False for k in grid})}
    ws2 = {"iso_a": _cell({k: True for k in grid}), "grid_b": _cell({k: True for k in grid})}
    ws2c = {"iso_a": _cell({k: True for k in grid})}  # grid_b has NO control arm

    def fake_load(directory, episodes, prefix):
        del episodes
        return ({"ws1": ws1, "ws2": ws2, "ws2c": ws2c}[prefix], grid)

    monkeypatch.setattr(mod, "load_journals", fake_load)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"segments": {"ws2c": {"cells": ["iso_a", "grid_b"]}}}))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"iso_a": {}, "grid_b": {}}))

    args = argparse.Namespace(
        ws1_dir=str(tmp_path), ws2_dir=str(tmp_path), ws2c_dir=str(tmp_path),
        manifest=str(manifest), episodes=2, tasks=1, index=str(index),
        allow_partial=True, resamples=50, seed=12345, csv="",
    )
    with pytest.raises(SystemExit, match="missing pairing"):
        mod.cmd_compare(args)


def test_full_matrix_table_runs_without_a_manifest(tmp_path, monkeypatch, capsys):
    import argparse

    from exp.robocasa365 import analyze_ws2_vs_ws1 as mod

    grid = [("OpenDrawer", 0), ("OpenDrawer", 1)]
    ws1 = {"grid_b": _cell({k: False for k in grid})}
    ws2 = {"grid_b": _cell({k: True for k in grid})}
    monkeypatch.setattr(mod, "load_journals", lambda d, e, p: ({"ws1": ws1, "ws2": ws2}[p], grid))

    index = tmp_path / "index.json"
    index.write_text(json.dumps({"grid_b": {}}))
    args = argparse.Namespace(
        ws1_dir=str(tmp_path), ws2_dir=str(tmp_path), ws2c_dir="",
        manifest="", episodes=2, tasks=1, index=str(index), allow_partial=False,
        resamples=50, seed=12345, csv="",
    )
    mod.cmd_compare(args)
    out = capsys.readouterr().out
    assert "JOINT effect" in out
    # The joint table must never advertise a factor split.
    assert "lib_effect" not in out
    assert "grid_b, 0.000, 1.000, +1.000" in out


# ------------------------------------------------------------------
# replay env fidelity (must equal the real eval env)
# ------------------------------------------------------------------


def test_replay_kwargs_come_from_the_production_adapter():
    """A second copy of the render resolution could drift from the eval env."""
    from exp.robocasa365.build_bucket_variants import eval_env_kwargs
    from exp.robocasa365.episode_runner import ADAPTERS

    assert eval_env_kwargs("groot_tp") == ADAPTERS["groot_tp"]().env_kwargs()
    # GR00T renders at its own resolution; the replay must carry it.
    assert set(eval_env_kwargs("groot_tp")) == {"camera_heights", "camera_widths"}
    # pi0.5 deliberately passes none (frozen against its admission gate).
    assert eval_env_kwargs("pi05") == {}


def test_unknown_teacher_is_refused():
    from exp.robocasa365.build_bucket_variants import eval_env_kwargs

    with pytest.raises(SystemExit, match="unknown teacher"):
        eval_env_kwargs("not_a_teacher")


def test_replay_builds_the_env_with_the_adapter_kwargs(monkeypatch):
    """Captures the ACTUAL gym.make call, not the declared intent."""
    import exp.robocasa365.build_bucket_variants as mod
    from exp.robocasa365.episode_runner import ADAPTERS

    calls = []

    class _Env:
        def reset(self, seed=None):
            return {"annotation.human.task_description": f"prompt for {seed}"}, {}

        def close(self):
            return None

    def fake_make(task_name, layout, style, **kwargs):
        calls.append((task_name, layout, style, kwargs))
        return _Env()

    monkeypatch.setattr("exp.robocasa365.episode_runner.default_gym_make", fake_make)
    monkeypatch.setattr("exp.robocasa365.episode_runner.PROMPT_SOURCE_KEY",
                        "annotation.human.task_description")
    buckets = [{"representative": {"trajectory": "OpenDrawer/episode_0007_a01",
                                   "seed": 7, "prompt": None, "status": "skipped"}}]
    mod.resolve_prompts(buckets, layout=1, style=1, teacher="groot_tp")

    assert len(calls) == 1
    task_name, layout, style, kwargs = calls[0]
    assert (task_name, layout, style) == ("OpenDrawer", 1, 1)
    assert kwargs == ADAPTERS["groot_tp"]().env_kwargs(), "replay env != eval env"
    assert buckets[0]["representative"]["prompt"] == "prompt for 7"
    assert buckets[0]["representative"]["status"] == "resolved"


def test_provenance_records_the_same_kwargs_it_replayed(monkeypatch):
    import exp.robocasa365.build_bucket_variants as mod
    from exp.robocasa365.episode_runner import ADAPTERS

    fake_robocasa = type("m", (), {"__file__": "/fake/robocasa/__init__.py", "__version__": "x"})
    monkeypatch.setitem(__import__("sys").modules, "robocasa", fake_robocasa)
    prov = mod.env_provenance(1, 1, "groot_tp")
    for key, value in ADAPTERS["groot_tp"]().env_kwargs().items():
        assert prov["env_kwargs"][key] == value
    assert prov["teacher"] == "groot_tp"
    assert "robocasa_commit" in prov


def test_bucket_key_matches_the_backend_on_a_real_pickled_artifact(tmp_path):
    """Bucket ids must be the ones the SERVED index uses, not the raw bytes.

    Artifacts store fp16 numpy; the backend rehydrates with
    ``torch.from_numpy(v).float()`` before keying. A tool reading the pickle
    directly sees the numpy arrays, so this drives both paths over the same
    file and demands identical grouping.
    """
    import pickle

    from openpi.cache.backends.in_memory_backend import InMemoryBackend
    from openpi.cache.config import TextIvfIndexConfig
    from openpi.cache.storage_types import CacheEntry, CachePayload, CheckpointID

    from exp.robocasa365.build_bucket_variants import bucket_entries

    import numpy as np

    def entry(eid: str, prompt: np.ndarray) -> CacheEntry:
        return CacheEntry(
            id=eid, checkpoint_id=CheckpointID.CP1,
            query_keys={"prompt_emb": prompt, "vision_0": np.zeros(2, dtype=np.float16)},
            payload=CachePayload(action_chunk=np.zeros((2, 2), dtype=np.float16)),
        )

    a = np.array([1.0, 0.0], dtype=np.float16)
    b = np.array([0.0, 1.0], dtype=np.float16)
    entries = [
        entry("OpenDrawer/episode_0000_a01:0", a),
        entry("OpenDrawer/episode_0000_a01:1", a),
        entry("CloseFridge/episode_0003_a01:0", b),
    ]
    path = tmp_path / "artifact.pkl"
    path.write_bytes(pickle.dumps({
        "vector_dims": {"prompt_emb": 2, "vision_0": 2},
        "entries": entries,
        "checkpoint_id": "CP1",
        "key_builder_type": "cp1_groot_spatial_pool_16",
        "prompt_pool": {"masked": False, "instruction_span": False},
    }))

    backend = InMemoryBackend(
        vector_dims={"prompt_emb": 2, "vision_0": 2},
        text_ivf=TextIvfIndexConfig(field="prompt_emb", max_buckets=16),
    )
    backend.load_artifact(str(path))
    served_buckets, _, _, _ = backend._text_ivf_state  # noqa: SLF001 - contract under test
    served = {frozenset(ids) for ids in served_buckets.values()}

    with path.open("rb") as fh:
        tool_entries = pickle.load(fh)["entries"]
    mapped = bucket_entries(tool_entries)
    tool = {frozenset(t for t in bucket["trajectories"]) for bucket in mapped}

    # Grouping must agree at trajectory level (ids differ only by step suffix).
    served_trajectories = {frozenset(i.rsplit(":", 1)[0] for i in ids) for ids in served_buckets.values()}
    assert tool == served_trajectories
    assert len(mapped) == len(served) == 2
