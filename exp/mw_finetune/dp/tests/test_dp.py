import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from exp.mw_finetune.dp.checkpoint import read_metadata
from exp.mw_finetune.dp.data import Batches, Dataset
from exp.mw_finetune.dp.model import DiffusionPolicy, crop
from exp.mw_finetune.dp.preprocess import quantile_normalize, quantile_unnormalize
from exp.mw_finetune.dp.train import learning_rate, newest_complete
from exp.mw_finetune.dp.policy import load_policy

ROOT = Path("/data/mw_finetune/dp")


@pytest.mark.parametrize("domain,dim", [("libero", 7), ("metaworld", 4), ("metaworld30k", 4)])
def test_real_checkpoint_strict_tree(domain, dim):
    root = ROOT / "checkpoints" / domain
    if not root.exists():
        pytest.skip("Run setup.sh for checkpoint integration tests")
    model = DiffusionPolicy(read_metadata(root)["config"]["model"])
    result = model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
    assert result.missing_keys == result.unexpected_keys == []
    assert len(model.state_dict()) == 475
    assert model.cfg["action_dim"] == dim
    assert not any("history" in key for key in model.state_dict())


def test_quantile_roundtrip_and_physical_action_range():
    stats = {"q01": [-7, -5, -13, -1], "q99": [10, 7, 24, 1]}
    x = np.array([[8, -4, 20, .1]], np.float32)
    np.testing.assert_allclose(quantile_unnormalize(quantile_normalize(x, stats), stats), x, atol=2e-6)
    assert quantile_unnormalize(np.ones((16, 4)), stats)[0, 2] > 1


def test_sampler_resume_independent_of_prefetch():
    whole = list(Batches(100, 42, 0, 5, 4, 2))
    resumed = list(Batches(100, 42, 3, 5, 4, 2))
    assert whole[6:] == resumed
    assert whole != list(Batches(100, 7, 0, 5, 4, 2))


def test_lr_original_schedule():
    assert learning_rate(0) < learning_rate(499)
    assert learning_rate(500) == pytest.approx(1e-4)
    assert learning_rate(100000) == pytest.approx(1e-5)
    assert learning_rate(99500) == pytest.approx(1e-5)


def test_crop_training_and_eval_use_only_current_image():
    x = torch.arange(96 * 96).reshape(1, 1, 96, 96).float().expand(2, 3, -1, -1)
    assert torch.equal(crop(x, False), x[..., 6:90, 6:90])
    assert crop(x, True).shape == (2, 3, 84, 84)


def test_real_dataset_episode_tail_and_no_history():
    root = ROOT / "mt50_96"
    if not (root / "manifest.json").exists():
        pytest.skip("Prepare full96 data for dataset integration test")
    data = Dataset(root, torch.zeros(50, 768))
    assert len(data) == 204806
    last = int(data.end_for_frame[0]) - 1
    sample = data[last]
    assert sample["obs"]["base_0_rgb"].shape == (3, 96, 96)
    assert sample["action"].shape == (16, 4)
    assert torch.equal(sample["action"][0].expand(16, 4), sample["action"])
    assert torch.count_nonzero(sample["obs"]["left_wrist_0_rgb"]) == 0
    assert len(data.manifest["tasks"]) == 50


def test_seedless_libero_does_not_read_advancing_global_rng():
    if not (ROOT / "clip_text/model.safetensors").exists():
        pytest.skip("Run setup.sh for real serving integration")
    policy = load_policy(ROOT / "checkpoints/libero", "cpu", domain="libero")
    assert policy.cuda_graph is False
    assert not hasattr(policy, "_denoiser_graph")
    assert "cuda_graph" not in policy.get_metadata()
    obs = {"observation/image": np.zeros((96, 96, 3), np.uint8),
           "observation/wrist_image": np.zeros((96, 96, 3), np.uint8),
           "observation/state": np.asarray(policy.stats["state"]["mean"], np.float32),
           "prompt": "pick up the black bowl and place it on the plate"}
    first = policy.infer(obs)["actions"]
    torch.manual_seed(9987)
    torch.randn(100)
    policy.reset()
    second = policy.infer(obs)["actions"]
    assert np.array_equal(first, second)
    assert policy.steps == 100


def test_real_libero_paired_pixels_vectors_and_episode_tail():
    root = ROOT / "libero_probe_96"
    if not (root / "manifest.json").exists():
        pytest.skip("Prepare one LIBERO source episode for integration test")
    import pyarrow.parquet as pq
    import io
    from PIL import Image
    data = Dataset(root, torch.zeros(40, 768), allow_partial=True)
    source = Path(data.manifest["source"]) / "data/chunk-000/episode_000000.parquet"
    rows = pq.read_table(source).to_pylist()
    sample = data[0]
    for raw, key in [("image", "base_0_rgb"), ("wrist_image", "left_wrist_0_rgb")]:
        with Image.open(io.BytesIO(rows[0][raw]["bytes"])) as image:
            expected = np.asarray(image.convert("RGB").resize((96,96), Image.Resampling.BILINEAR))
        got = (sample["obs"][key].permute(1,2,0).numpy() * 255).round().astype(np.uint8)
        assert np.array_equal(got, expected)
    state = quantile_unnormalize(sample["obs"]["state"].numpy(), data.manifest["external_stats"]["state"])
    np.testing.assert_allclose(state, rows[0]["state"], atol=1e-6)
    last = data[len(rows)-1]["action"]
    assert last.shape == (16,7) and torch.equal(last, last[0].expand(16,7))
    with pytest.raises(ValueError): Dataset(root, torch.zeros(40,768))


def test_resume_recovers_complete_save_when_latest_update_was_interrupted(tmp_path):
    for step in (1,2):
        path=tmp_path/f"checkpoint-{step:06d}";path.mkdir()
        (path/"complete.json").write_text(json.dumps({"step":step}))
        for name in ("model.safetensors","training.pt","metadata.json","norm_stats.json"):
            (path/name).touch()
    (tmp_path/"latest").symlink_to("checkpoint-000001")
    (tmp_path/".checkpoint-000003.partial").mkdir()
    assert newest_complete(tmp_path).name == "checkpoint-000002"
