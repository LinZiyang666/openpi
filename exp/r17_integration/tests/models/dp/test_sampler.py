from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from diffusers import DDIMScheduler

from exp.mw_finetune.dp.model import DiffusionPolicy
from exp.r17_integration.models.dp.sampler import SamplerProfile, checked_profile


def audited_config():
    return json.loads((Path(__file__).parents[3] / "models/dp/samples/audited_libero_metadata.json").read_text())["config"]["model"]


def test_default_ddpm100_matches_E_byte_exact():
    cfg = audited_config()
    instance = type("Config", (), {"cfg": cfg})()
    expected = DiffusionPolicy.scheduler(instance, 100)
    got = SamplerProfile().scheduler(cfg)
    assert torch.equal(expected.timesteps, got.timesteps) and torch.equal(expected.betas, got.betas)
    assert tuple(int(t) for t in got.timesteps) == tuple(range(99, -1, -1))
    assert SamplerProfile().kind == "ddpm"


def test_every_reduced_K_starts_at_Tminus1_and_binds_trailing_identity():
    cfg = audited_config()
    for k in range(1, 101):
        profile = SamplerProfile("ddim", k)
        spec = profile.to_spec(cfg)
        assert spec.native_timesteps[0] == 99
        assert len(set(spec.native_timesteps)) == k
        assert all(a > b for a, b in zip(spec.native_timesteps, spec.native_timesteps[1:]))
        assert spec.config["timestep_spacing"] == "trailing"
        assert spec.config["config"]["timestep_spacing"] == "trailing"
        assert spec.config["update_rule"] == "native_grid_predecessor_ddim_v1"
    assert SamplerProfile(num_steps=1).to_spec(cfg).native_timesteps == (99,)
    assert SamplerProfile(num_steps=2).to_spec(cfg).native_timesteps == (99, 49)
    assert SamplerProfile(num_steps=4).to_spec(cfg).native_timesteps == (99, 74, 49, 24)


def test_ddim_identity_eta_and_old_leading_grid_rejection():
    cfg = audited_config()
    baseline = SamplerProfile().to_spec(cfg)
    ddim = SamplerProfile("ddim", 2).to_spec(cfg)
    noisy = SamplerProfile("ddim", 2, 0.5).to_spec(cfg)
    assert len({baseline.identity, ddim.identity, noisy.identity}) == 3
    assert ddim.config["eta"] == 0 and noisy.config["eta"] == 0.5
    assert "variance_type" not in ddim.config["config"]
    for bad in [replace(ddim, native_timesteps=(50, 0)),
                replace(ddim, config={**ddim.config, "timestep_spacing": "leading"}),
                replace(ddim, config={**ddim.config, "eta": .2})]:
        with pytest.raises(ValueError):
            checked_profile(bad, cfg)


@pytest.mark.parametrize("k,eta", [(1, 0), (2, .5), (4, 0), (10, .5), (100, 0)])
def test_ddim_divisor_updates_match_diffusers_and_private_noise(k, eta):
    profile = SamplerProfile("ddim", k, eta)
    cfg = audited_config()
    scheduler = profile.scheduler(cfg)
    reference = DDIMScheduler.from_config(scheduler.config)
    reference.set_timesteps(k)
    x = torch.linspace(-1, 1, 64).reshape(1,16,4)
    a, b = torch.Generator().manual_seed(19), torch.Generator().manual_seed(19)
    for t in scheduler.timesteps:
        prediction = x * .2
        expected = reference.step(prediction, t, x, eta=eta, generator=a).prev_sample
        got = profile.step(scheduler, prediction, t, x, b)
        assert torch.equal(got, expected)
        assert torch.equal(a.get_state(), b.get_state())
        x = got


@pytest.mark.parametrize("k", [3, 7, 13, 99])
def test_nondivisor_ddim_uses_actual_next_grid_alpha(k):
    profile = SamplerProfile("ddim", k)
    scheduler = profile.scheduler(audited_config())
    x = torch.linspace(-.7, .7, 64).reshape(1,16,4)
    prediction = .2*x
    for i, t in enumerate(scheduler.timesteps):
        previous = scheduler.timesteps[i+1] if i+1 < k else -1
        a = scheduler.alphas_cumprod[t]
        ap = scheduler.alphas_cumprod[previous] if previous >= 0 else scheduler.final_alpha_cumprod
        clean = ((x - (1-a).sqrt()*prediction)/a.sqrt()).clamp(-1,1)
        expected = ap.sqrt()*clean + (1-ap).sqrt()*prediction
        assert torch.equal(profile.step(scheduler, prediction, t, x, torch.Generator()), expected)


def test_reviewer_oracle_K1_does_not_treat_pure_noise_as_t0_and_is_labelled_lossy():
    cfg = audited_config()
    profile = SamplerProfile(num_steps=1)
    scheduler = profile.scheduler(cfg)
    t = scheduler.timesteps[0]
    x = torch.randn((10000,4), generator=torch.Generator().manual_seed(0))
    alpha = scheduler.alphas_cumprod[t]
    epsilon = (1-alpha).sqrt()*(x-alpha.sqrt()*.4)/(alpha*.2**2+1-alpha)
    got = profile.step(scheduler, epsilon, t, x, torch.Generator())
    assert abs(float(got.mean()) - .4) < .005
    # One epsilon step cannot reproduce the target variance; the label is
    # deliberately lossy even though the grid/mathematical update is valid.
    assert float(got.std()) < .01
    assert profile.to_spec(cfg).config["quality_status"] == "lossy"


@pytest.mark.parametrize("args", [("flow", 2), ("ddpm", 0), ("ddpm", 101), ("ddpm", True),
                                 ("ddpm", 1.5), ("ddpm", 2), ("ddpm", 2, .5),
                                 ("ddim", 2, -1), ("ddim", 2, 0, "leading")])
def test_sampler_rejects_invalid_or_flow_configs(args):
    with pytest.raises(ValueError):
        SamplerProfile(*args)
