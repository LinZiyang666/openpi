"""Manual tests for the RoboCasa365 DataConfig -- require the GR00T island.

Kept in a file of its own, apart from the non-manual adapter tests, because
``manual`` only skips *after* the module has been imported: a top-level
``import gr00t`` would make collection itself fail in the main environment, long
before the marker is consulted.  Hence both guards below -- the module-level
``importorskip`` and imports confined to function bodies.

These are model-free: they exercise the DataConfig contract and the rotation
convention, and none of them loads the 7.2 GB checkpoint weights.  Only the
checkpoint's ``experiment_cfg/metadata.json`` is read, so the suite runs in
seconds.  GPU/simulation checks (G0-A wire parity, rollouts) stay out of here.

Run inside the GR00T island.  Three things are easy to get wrong here:

* ``gr00t`` is **not installed** in that venv -- it is imported from the
  ``n1.5-release`` worktree at ``/home/weiland/gr00t_n15`` via ``PYTHONPATH``.
* ``pytest`` and ``decord`` are not in that venv by default.  ``decord`` is
  pulled in indirectly when the parent DataConfig builds its transform chain, so
  without it every DataConfig test dies on an ImportError.
* the repo's ``conftest.py`` default-skips ``manual``; ``--run-manual`` is the
  opt-in (``-m manual`` alone only *selects* them, they still skip).

::

    /home/weiland/gr00t_n15_venv/.venv/bin/python -m pip install pytest decord
    cd /home/weiland/openpi && \\
    PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \\
      /home/weiland/gr00t_n15_venv/.venv/bin/python -m pytest \\
      tests/robocasa365/test_groot_data_config_manual.py --run-manual -q

Point ``ROBOCASA365_N15_CHECKPOINT`` at the checkpoint if it is not at the
default path recorded below.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from exp.robocasa365 import groot_keys

DEFAULT_CHECKPOINT = pathlib.Path(
    "/home/weiland/ckpt_n15_robocasa/gr00t_n1-5/multitask_learning/checkpoint-120000"
)
EMBODIMENT_TAG = "new_embodiment"

pytest.importorskip("gr00t", reason="GR00T is only installed in the GR00T island")

pytestmark = pytest.mark.manual


def test_data_config_matches_official_ordered_config():
    """The four ordered lists must survive construction unchanged."""
    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    cfg = RoboCasa365DataConfig()
    assert cfg.video_keys == groot_keys.VIDEO_KEYS
    assert cfg.state_keys == groot_keys.STATE_KEYS
    assert cfg.action_keys == groot_keys.ACTION_KEYS
    assert cfg.language_keys == groot_keys.LANGUAGE_KEYS
    assert len(cfg.action_indices) == groot_keys.ACTION_HORIZON


def test_language_key_overrides_parent():
    """The parent's N1.7 variant must not leak through by inheritance."""
    from gr00t.experiment.data_config import SinglePandaGripperDataConfig

    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    assert (
        RoboCasa365DataConfig().language_keys
        != SinglePandaGripperDataConfig().language_keys
    )
    assert RoboCasa365DataConfig().language_keys == [
        "annotation.human.task_description"
    ]


def test_modality_config_keys_and_order():
    """modality_config() is what actually drives ConcatTransform."""
    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    modality = RoboCasa365DataConfig().modality_config()
    assert list(modality["video"].modality_keys) == groot_keys.VIDEO_KEYS
    assert list(modality["state"].modality_keys) == groot_keys.STATE_KEYS
    assert list(modality["action"].modality_keys) == groot_keys.ACTION_KEYS
    assert list(modality["language"].modality_keys) == groot_keys.LANGUAGE_KEYS


def test_unit_quaternion_survives_rotation_transform():
    """The identity quaternion must map to a finite rotation_6d vector.

    Confirms the wxyz convention empirically; the all-zeros alternative would
    produce non-finite output here.
    """
    import numpy as np
    import torch
    from pytorch3d.transforms import quaternion_to_matrix

    identity = torch.tensor(groot_keys.IDENTITY_QUATERNION_WXYZ, dtype=torch.float32)
    matrix = quaternion_to_matrix(identity)
    assert torch.allclose(matrix, torch.eye(3), atol=1e-6)
    assert np.isfinite(matrix.numpy()).all()


def _checkpoint_metadata() -> dict:
    """Load the checkpoint's metadata, skipping if the weights are not present."""
    root = pathlib.Path(
        os.environ.get("ROBOCASA365_N15_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    metadata = root / "experiment_cfg" / "metadata.json"
    if not metadata.exists():
        pytest.skip(f"checkpoint metadata not found at {metadata}")
    return json.loads(metadata.read_text())[EMBODIMENT_TAG]


def test_modality_config_covers_checkpoint_keys():
    """Every declared key must exist in the checkpoint, and vice versa.

    Compared as sets: the metadata is alphabetically sorted and carries no
    ordering information, so order is asserted elsewhere against the official
    config instead.
    """
    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    meta = _checkpoint_metadata()["modalities"]
    modality = RoboCasa365DataConfig().modality_config()

    for name, prefix in (
        ("video", "video."),
        ("state", "state."),
        ("action", "action."),
    ):
        declared = {key.removeprefix(prefix) for key in modality[name].modality_keys}
        assert declared == set(meta[name]), f"{name} key set differs from checkpoint"


def test_state_and_action_dims_match_checkpoint():
    """The per-key widths in groot_keys must match the trained statistics."""
    stats = _checkpoint_metadata()["statistics"]

    for key, width in groot_keys.STATE_DIMS.items():
        mean = stats["state"][key.removeprefix("state.")]["mean"]
        assert len(mean) == width, f"{key}: declared {width}, checkpoint {len(mean)}"

    for key, width in groot_keys.ACTION_DIMS.items():
        mean = stats["action"][key.removeprefix("action.")]["mean"]
        assert len(mean) == width, f"{key}: declared {width}, checkpoint {len(mean)}"


def test_checkpoint_embodiment_tag_is_new_embodiment():
    assert _checkpoint_metadata()["embodiment_tag"] == EMBODIMENT_TAG


def test_transform_chain_constructs():
    """Building transform() also proves its runtime deps (decord) are present."""
    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    transform = RoboCasa365DataConfig().transform()
    assert transform is not None


def test_transform_concat_order_uses_our_keys():
    """The strongest check available without a GPU.

    ``modality_config()`` is self-certifying: the handshake compares the very
    object it was handed.  ``ConcatTransform`` is what actually splices the
    per-key tensors, so if the parent built it from hardcoded lists instead of
    ``self.*_keys``, our overrides would apply to the config but not to the
    tensors -- and every existing check, handshake included, would still report
    a tidy match.  This is the one place where being wrong passes all the gates.
    """
    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig

    transform = RoboCasa365DataConfig().transform()
    concat = [t for t in transform.transforms if type(t).__name__ == "ConcatTransform"]
    assert concat, (
        f"no ConcatTransform in {[type(t).__name__ for t in transform.transforms]}"
    )
    node = concat[0]
    assert list(node.video_concat_order) == groot_keys.VIDEO_KEYS
    assert list(node.state_concat_order) == groot_keys.STATE_KEYS
    assert list(node.action_concat_order) == groot_keys.ACTION_KEYS


def test_statistics_have_matching_mean_and_std_widths():
    """A truncated or padded stats block is exactly what this should catch."""
    stats = _checkpoint_metadata()["statistics"]
    for group, dims in (
        ("state", groot_keys.STATE_DIMS),
        ("action", groot_keys.ACTION_DIMS),
    ):
        for key, width in dims.items():
            entry = stats[group][key.split(".", 1)[1]]
            assert len(entry["mean"]) == width
            assert len(entry["std"]) == width
