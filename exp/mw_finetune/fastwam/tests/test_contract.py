"""CPU checks for irreversible mistakes in data boundaries and action conventions."""
import json
import io
import sqlite3
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from exp.mw_finetune.fastwam.common import denormalize, lr_at, normalize, window_indices
from exp.mw_finetune.fastwam.data import Dataset
from exp.mw_finetune.fastwam.model import canonical_key, load_delta, save_delta
from exp.mw_finetune.fastwam.policy import Policy


STATS = {"min": [-12, -15, -18, -1], "max": [20, 13, 25, 1]}


def test_normalization_returns_unclipped_physical_commands():
    actions = np.array([[19, -14, 24, -0.25], [-20, 20, 30, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(denormalize(normalize(actions, STATS), STATS), actions, atol=2e-6)
    assert abs(normalize(actions, STATS)[1, 0]) > 1


def test_boundaries_and_scheduler():
    action, mask, images, image_mask = window_indices(3, 5)
    np.testing.assert_array_equal(action[:4], [3, 4, 4, 4])
    assert mask.sum() == 30
    assert images.max() == 4 and image_mask.sum() == 8
    assert lr_at(0) == pytest.approx(1e-6)
    assert lr_at(99) == pytest.approx(1e-4)
    assert lr_at(100) == pytest.approx(1e-4)
    assert lr_at(99999) == pytest.approx(2.5e-6)


def test_reader_never_uses_next_episode(tmp_path):
    frames = 4
    np.savez(tmp_path / "index.npz", index=np.arange(frames), part=np.zeros(frames, np.int32),
        episode_index=np.array([0, 0, 1, 1]), frame_index=np.array([0, 1, 0, 1]),
        task_index=np.zeros(frames, np.int64), state=np.zeros((frames, 4), np.float32),
        action=np.array([[0]*4, [1]*4, [10]*4, [11]*4], np.float32))
    manifest = {"partial": False, "shards": ["part.sqlite"], "tasks": {"0": "reach"},
        "stats": {"action": STATS, "observation.state": STATS}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with sqlite3.connect(tmp_path / "part.sqlite") as db:
        db.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, png BLOB)")
        for i in range(frames):
            buf = io.BytesIO()
            Image.fromarray(np.full((224, 224, 3), i*50, np.uint8)).save(buf, format="PNG")
            db.execute("INSERT INTO images VALUES (?,?)", (i, buf.getvalue()))
    sample = Dataset(tmp_path)[1]
    recovered = denormalize(sample["action"].numpy(), STATS)
    np.testing.assert_allclose(recovered, 1, atol=1e-6)
    assert sample["action_is_pad"].sum() == 31
    np.testing.assert_allclose(sample["video"].numpy(), 50/255, atol=1e-7)


class Core:
    def infer_action(self, **kw):
        self.call = kw
        generator = torch.Generator().manual_seed(kw["seed"])
        return {"action": torch.randn((1, 32, 4), generator=generator)}


def test_stateless_wrapper_seed_prompt_and_gripper():
    core = Core()
    reset = []
    p = Policy(SimpleNamespace(model=core, reset=lambda: reset.append(1),
        config=SimpleNamespace(prompt_template="instruction: {task}")),
        {"stats": {"action": STATS, "observation.state": STATS}})
    obs = {"image": np.full((480, 480, 3), 128, np.uint8), "state": np.zeros(4, np.float32), "prompt": "reach", "seed": 17}
    a = p.infer(obs)["actions"]
    p.reset()
    b = p.infer(obs)["actions"]
    np.testing.assert_array_equal(a, b)
    assert a.shape == (32, 4) and a.dtype == np.float32
    assert core.call["prompt"] == "instruction: reach"
    assert core.call["num_inference_steps"] == 10
    assert core.call["input_image"].shape == (1, 3, 224, 224)
    assert np.unique(a[:, 3]).size > 2
    obs["seed"] = 18
    assert not np.array_equal(a, p.infer(obs)["actions"])


def test_checkpoint_key_remap_and_strict_delta(tmp_path):
    assert canonical_key("model.dit.mixtures.action.blocks.3.ffn.0.weight") == "model.mot.layers.3.blocks.action.ffn.0.weight"
    module = torch.nn.Linear(4, 4)
    original = module.weight.detach().clone()
    save_delta(module, tmp_path, {"format": "test"})
    with torch.no_grad():
        module.weight.zero_()
    load_delta(module, tmp_path)
    torch.testing.assert_close(module.weight, original)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_delta(torch.nn.Linear(3, 4), tmp_path)
