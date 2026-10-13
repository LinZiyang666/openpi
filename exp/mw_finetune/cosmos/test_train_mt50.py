"""CPU regression checks for MT50 targets and resumable training."""
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from .mt50_data import FrameSampler, actions, canonical_prompts, dataset_statistics, normalize, proprio
from .train_mt50 import fit, optimizer_restore, parse_args


def test_our_statistics_and_fixed_channels_round_trip():
    state = np.array([[.1, .5, .2, .3], [.4, .8, .4, 1]], np.float32)
    action = np.array([[-2, 3, 4, -1], [5, -6, 7, 1]], np.float32)
    stats = dataset_statistics(dict(state=state, action=action))
    for name, raw in (("proprio", proprio(state)), ("actions", actions(action))):
        scaled = normalize(raw, stats, name)
        restored = (scaled + 1) * (np.asarray(stats[name + "_max"]) - stats[name + "_min"]) / 2 + stats[name + "_min"]
        np.testing.assert_allclose(restored, raw, atol=1e-7)
        assert np.isfinite(scaled).all()
    assert np.all(normalize(actions(action), stats, "actions")[:, 3:6] == 0)
    assert np.all(normalize(proprio(state), stats, "proprio")[:, 5:] == 0)
    np.testing.assert_array_equal(actions(action)[:, [0, 1, 2, 6]], action)


def test_sampler_resume_and_epoch_boundaries():
    full = list(FrameSampler(13, 7, 0, 39))
    resumed = list(FrameSampler(13, 7, 11, 28))
    assert full[11:] == resumed
    for start in (0, 13, 26):
        assert sorted(frame for frame, _ in full[start:start + 13]) == list(range(13))
    assert len(list(FrameSampler(13, 7, 0, 0))) == 0


def test_prompt_indices_are_labels_not_r17_task_positions():
    result = canonical_prompts({42: "Reach a goal position", 49: "Push the puck back toward the robot to the goal"})
    assert result == {"42": "Reach a goal position", "49": "Push the puck to a goal"}


def test_master_weights_and_moment_precision_survive_restore():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    opt = torch.optim.AdamW([p])
    saved = opt.state_dict()
    original = torch.tensor([1.00001, .1234567], dtype=torch.float32)
    saved["state"][0] = dict(step=torch.tensor(1.), exp_avg=original, exp_avg_sq=original * .9)
    optimizer_restore(opt, saved, [[original]], "cpu")
    assert torch.equal(opt.state[p]["exp_avg"], original)
    assert opt.state[p]["exp_avg"].dtype == torch.float32
    assert torch.equal(opt.param_groups_master[0]["params"][0], original)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def training_step(self, batch, iteration):
        loss = ((self.weight * batch["x"] + torch.rand(()) - batch["y"]) ** 2).mean()
        return {"demo_sample_action_l1_loss": loss.sqrt()}, loss


class TinyDataset:
    def __len__(self):
        return 13

    def __getitem__(self, pair):
        frame, _ = pair
        return dict(x=torch.tensor(float(frame + 1)), y=torch.tensor(.1))


def args(tmp_path, **kw):
    assets = tmp_path / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("dataset_statistics.json", "t5_embeddings.pkl"):
        (assets / name).write_text("cpu-test")
    a = parse_args(["--output", str(tmp_path / "train"), "--assets", str(assets), "--workers", "0",
        "--steps", "4", "--batch", "2", "--accum", "2", "--save-every", "2", "--export-every", "4"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def setup():
    model = TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.9)
    return model, opt, schedule


def test_bounded_probe_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(7)
    (tmp_path / "full").mkdir()
    a = args(tmp_path / "full")
    m, o, s = setup()
    report = fit(a, m, o, s, TinyDataset(), device="cpu")
    reference = m.weight.detach().clone()
    assert report["completed_updates"] == 4
    torch.manual_seed(7)
    (tmp_path / "split").mkdir()
    a = args(tmp_path / "split", probe=2)
    m, o, s = setup()
    assert fit(a, m, o, s, TinyDataset(), device="cpu")["completed_updates"] == 2
    m, o, s = setup()
    a.probe = 0
    assert fit(a, m, o, s, TinyDataset(), device="cpu")["completed_updates"] == 2
    assert torch.equal(m.weight.detach(), reference)
    exported = json.loads((Path(a.output) / "checkpoint-000004/config.json").read_text())
    assert exported["cosmos_mw"]["use_jpeg_compression"] is False
    assert exported["training"]["completed_steps"] == 4


def test_expired_bound_starts_no_update(tmp_path):
    a = args(tmp_path)
    m, o, s = setup()
    import time
    report = fit(a, m, o, s, TinyDataset(), device="cpu", started=time.monotonic() - 25 * 3600)
    assert report["completed_updates"] == 0 and report["stop_reason"] == "max_hours"
    assert not (Path(a.output) / "training_state.pt").exists()


def test_resume_rejects_changed_effective_batch(tmp_path):
    a = args(tmp_path, probe=1)
    m, o, s = setup()
    fit(a, m, o, s, TinyDataset(), device="cpu")
    a.batch = 3
    m, o, s = setup()
    with pytest.raises(ValueError, match="effective batch"):
        fit(a, m, o, s, TinyDataset(), device="cpu")


def test_print_config_and_invalid_bounds(capsys):
    from .train_mt50 import main
    main(["--output", "/unused", "--print-config"])
    assert json.loads(capsys.readouterr().out)["effective_batch"] == 32
    with pytest.raises(SystemExit):
        parse_args(["--output", "/unused", "--max-hours", "0"])


def test_native_load_disables_inference_reseeding_and_keeps_vae_separate(monkeypatch, tmp_path):
    from .train_mt50 import load_native
    monkeypatch.setattr('exp.r17_integration.models.cosmos.loader.activate_vendor',
                        lambda _: monkeypatch.setenv('DETERMINISTIC', 'True'))
    vae = tmp_path / 'vae.pth'
    monkeypatch.setattr('exp.r17_integration.models.cosmos.loader.tokenizer_path', lambda _: vae)
    names = ['cosmos_policy', 'cosmos_policy._src', 'cosmos_policy._src.imaginaire',
        'cosmos_policy._src.imaginaire.utils', 'cosmos_policy._src.imaginaire.utils.checkpoint_db',
        'cosmos_policy._src.predict2', 'cosmos_policy._src.predict2.utils',
        'cosmos_policy._src.predict2.utils.model_loader']
    modules = {}
    for name in names:
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(modules[parent], attr, module)
    db = modules['cosmos_policy._src.imaginaire.utils.checkpoint_db']
    db.get_checkpoint_path = lambda value: value
    def fake_load(**kw):
        assert os.environ['DETERMINISTIC'] == 'False'
        assert db.get_checkpoint_path('hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth') == str(vae)
        assert kw['experiment_name'] == 'cosmos_predict2_2b_480p_libero'
        raise RuntimeError('CPU fake load reached')
    modules['cosmos_policy._src.predict2.utils.model_loader'].load_model_from_checkpoint = fake_load
    with pytest.raises(RuntimeError, match='CPU fake load reached'):
        load_native(parse_args(['--output', str(tmp_path)]))
