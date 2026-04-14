"""Tests for ``EpisodeDataCollector`` (plan §3.4 / §22.4).

Focus areas flagged by G2 §22.4:
- ``episode_name`` naming (subdir support + path-escape defence);
- ``set_episode_attr`` merging into HDF5 attrs;
- ``_episode_attrs`` lifecycle: initialised in ``__init__``, cleared in
  ``on_episode_start``, retained if set before the first episode starts;
- legacy timestamp naming when ``episode_name`` is empty.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings


def _dummy_embeddings() -> InferenceEmbeddings:
    """Smallest valid embedding record — 1 vision emb, 1 noise step."""
    return InferenceEmbeddings(
        vision_embs=[np.zeros((1, 4), dtype=np.float16)],
        prompt_emb=np.zeros((1, 4), dtype=np.float16),
        robot_state=np.zeros(8, dtype=np.float32),
        noise_action_steps=[np.zeros(7, dtype=np.float32)],
        clean_action=np.zeros(7, dtype=np.float32),
        input_images=None,
    )


# ---------------------------------------------------------------------------
# episode_name-driven naming
# ---------------------------------------------------------------------------


def test_episode_name_writes_under_subdir(tmp_path: Path) -> None:
    collector = EpisodeDataCollector(str(tmp_path))
    collector.on_episode_start(
        "exp", "task", 1, episode_name="task_3/episode_7"
    )
    collector.record_inference(_dummy_embeddings())
    collector.on_episode_end(success=True)

    expected = tmp_path / "exp" / "task_3" / "episode_7.h5"
    assert expected.exists()
    with h5py.File(expected, "r") as f:
        assert f.attrs["experiment_name"] == "exp"
        assert f.attrs["episode_id"] == 1


def test_legacy_timestamp_naming_when_episode_name_empty(tmp_path: Path) -> None:
    """Empty ``episode_name`` must preserve the legacy ``episode_{id}_{ts}.h5``
    filename so pre-Layer-B-4 pipelines keep working unchanged."""
    collector = EpisodeDataCollector(str(tmp_path))
    collector.on_episode_start("exp", "task", 42)
    collector.record_inference(_dummy_embeddings())
    collector.on_episode_end(success=False)

    files = list((tmp_path / "exp").glob("episode_0042_*.h5"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# Path-escape defence (plan §22 decision kept)
# ---------------------------------------------------------------------------


def test_episode_name_rejects_path_traversal(tmp_path: Path) -> None:
    collector = EpisodeDataCollector(str(tmp_path))
    collector.on_episode_start("exp", "task", 1, episode_name="../evil")
    collector.record_inference(_dummy_embeddings())

    with pytest.raises(ValueError, match="escapes base directory"):
        collector.on_episode_end(success=True)

    # And no file was written outside the base dir.
    assert not (tmp_path.parent / "evil.h5").exists()


# ---------------------------------------------------------------------------
# set_episode_attr
# ---------------------------------------------------------------------------


def test_set_episode_attr_written_to_hdf5(tmp_path: Path) -> None:
    collector = EpisodeDataCollector(str(tmp_path))
    collector.on_episode_start("exp", "task", 0, episode_name="ep_0")
    collector.set_episode_attr("prompt", "pick up the apple")
    collector.set_episode_attr("init_state_idx", 15)
    collector.record_inference(_dummy_embeddings())
    collector.on_episode_end(success=True)

    with h5py.File(tmp_path / "exp" / "ep_0.h5", "r") as f:
        assert f.attrs["prompt"] == "pick up the apple"
        assert int(f.attrs["init_state_idx"]) == 15


def test_on_episode_start_clears_previous_attrs(tmp_path: Path) -> None:
    """Per §22 decision: ``_episode_attrs`` is reused (cleared, not rebound)
    so attrs from episode A do not leak into episode B."""
    collector = EpisodeDataCollector(str(tmp_path))

    collector.on_episode_start("exp", "task", 0, episode_name="ep_0")
    collector.set_episode_attr("prompt", "first")
    collector.record_inference(_dummy_embeddings())
    collector.on_episode_end(success=True)

    collector.on_episode_start("exp", "task", 1, episode_name="ep_1")
    # Intentionally do NOT set prompt for ep_1.
    collector.record_inference(_dummy_embeddings())
    collector.on_episode_end(success=True)

    with h5py.File(tmp_path / "exp" / "ep_1.h5", "r") as f:
        assert "prompt" not in f.attrs


def test_set_episode_attr_before_episode_start_silently_accepted(
    tmp_path: Path,
) -> None:
    """Calls before ``on_episode_start`` must not raise (mirrors
    ``record_inference``'s tolerant contract). The attr survives until the
    next ``on_episode_start`` clears it, at which point the caller is
    expected to re-set it — this is fine because the canonical integration
    (``CollectionPolicy``) only ever calls ``set_episode_attr`` inside
    ``infer``, which runs after ``on_episode_start``."""
    collector = EpisodeDataCollector(str(tmp_path))
    # Must not raise.
    collector.set_episode_attr("stray", "value")

    # Now open an episode — stray attr should have been cleared.
    collector.on_episode_start("exp", "task", 0, episode_name="ep_0")
    collector.record_inference(_dummy_embeddings())
    collector.on_episode_end(success=True)

    with h5py.File(tmp_path / "exp" / "ep_0.h5", "r") as f:
        assert "stray" not in f.attrs
