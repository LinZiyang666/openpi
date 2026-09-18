"""CPU tests for exp/dp_nfe/dp_sampler.py: the trailing grid, the two-head DDIM update against a float64 reference, the
NFE count, the inpainting mask, the eps_mode switch, the keyed-noise hooks, the local/global conditioning pass-through
and the DDPM posterior step against an independent numpy transcription of diffusers 0.11.1 ``DDPMScheduler.step``
(fixed_small). Parity against the real diffusers scheduler lives in the env_dependent workspace tests."""

import math

import numpy as np
import pytest
import torch

from exp.dp_nfe import dp_sampler as S


def _cosine_alphas_cumprod(T=100, s=0.008):
    # diffusers "squaredcos_cap_v2": betas from the cosine schedule, capped at 0.999
    def f(t):
        return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    betas = [min(1 - f(i + 1) / f(i), 0.999) for i in range(T)]
    ac, cur = [], 1.0
    for b in betas:
        cur *= 1 - b
        ac.append(cur)
    return np.array(ac, dtype=np.float64)


@pytest.mark.parametrize("k,expected", [(1, [99]), (2, [99, 49]), (4, [99, 74, 49, 24]), (10, list(range(99, -1, -10))),
                                        (100, list(range(99, -1, -1)))])
def test_trailing_grid(k, expected):
    assert S.make_timesteps(100, k) == expected


@pytest.mark.parametrize("k", [3, 7, 50, 0, -1])
def test_grid_rejects_unregistered_k(k):
    with pytest.raises(ValueError):
        S.make_timesteps(100, k)


def test_heads_agree_on_consistent_predictions():
    """Given the same (x_t, t) and a consistent (eps, x0) pair, both heads produce the same x_prev (float64, atol 1e-8)."""
    ac = _cosine_alphas_cumprod()
    rng = np.random.default_rng(0)
    for t in (99, 49, 24, 5):
        x_t = torch.tensor(rng.normal(size=(2, 8, 3)) * 0.5)
        x0_true = torch.tensor(rng.uniform(-0.9, 0.9, size=(2, 8, 3)))
        a = ac[t]
        eps_true = (x_t - math.sqrt(a) * x0_true) / math.sqrt(1 - a)
        x0_e, eps_e = S.to_x0_eps(x_t, eps_true, a, "epsilon")
        x0_s, eps_s = S.to_x0_eps(x_t, x0_true, a, "sample")
        assert torch.allclose(x0_e, x0_s, atol=1e-8) and torch.allclose(eps_e, eps_s, atol=1e-8)
        for a_prev in (ac[t - 1] if t > 0 else 1.0, 1.0):
            assert torch.allclose(S.ddim_step(x0_e, eps_e, a_prev), S.ddim_step(x0_s, eps_s, a_prev), atol=1e-8)


def test_clip_recomputes_eps_and_last_step_returns_x0():
    ac = _cosine_alphas_cumprod()
    x_t = torch.zeros(1, 4, 2, dtype=torch.float64)
    pred = torch.full((1, 4, 2), 3.0, dtype=torch.float64)  # x0 head predicting out of range
    x0, eps = S.to_x0_eps(x_t, pred, ac[99], "sample", clip=True)
    assert torch.all(x0 == 1.0)
    ref_eps = (x_t - math.sqrt(ac[99]) * x0) / math.sqrt(1 - ac[99])
    assert torch.allclose(eps, ref_eps, atol=1e-12)
    assert torch.allclose(S.ddim_step(x0, eps, 1.0), x0)  # a_prev = 1 -> x0 exactly


def test_reference_loop_matches_numpy_float64():
    """Full k=4 loop against an independent numpy implementation of the same formulas."""
    ac = _cosine_alphas_cumprod()
    ts = S.make_timesteps(100, 4)
    rng = np.random.default_rng(1)
    W = rng.normal(size=(6, 6)) * 0.1
    def net_np(x, t):  # a fixed nonlinear "network" returning eps
        return np.tanh(x @ W + 0.01 * t)
    def net_t(x, t):
        return torch.tanh(x @ torch.tensor(W) + 0.01 * t)
    x = rng.normal(size=(3, 6))
    xt = torch.tensor(x)
    # numpy reference
    xr = x.copy()
    for i, t in enumerate(ts):
        a = ac[t]
        e = net_np(xr, t)
        x0 = np.clip((xr - math.sqrt(1 - a) * e) / math.sqrt(a), -1, 1)
        e2 = (xr - math.sqrt(a) * x0) / math.sqrt(1 - a)
        ap = ac[ts[i + 1]] if i + 1 < len(ts) else 1.0
        xr = math.sqrt(ap) * x0 + math.sqrt(1 - ap) * e2
    out, nfe = S.sample_ddim(net_t, xt, ts, ac, "epsilon", clip=True)
    assert nfe == 4
    assert np.allclose(out.numpy(), xr, atol=1e-10)
    assert np.all(np.abs(xr) <= 1.0 + 1e-12)


def test_inpaint_mask_is_applied_before_each_call_and_on_output():
    ac = _cosine_alphas_cumprod()
    ts = S.make_timesteps(100, 2)
    seen = []
    def net(x, t):
        seen.append(x.clone())
        return torch.zeros_like(x)
    data = torch.full((1, 4, 2), 0.25, dtype=torch.float64)
    mask = torch.zeros(1, 4, 2, dtype=torch.bool); mask[:, 0] = True
    x_T = torch.randn(1, 4, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    out, nfe = S.sample_ddim(net, x_T, ts, ac, "sample", inpaint=(data, mask))
    assert nfe == 2 and len(seen) == 2
    for s in seen:
        assert torch.all(s[:, 0] == 0.25)
    assert torch.all(out[:, 0] == 0.25)


class _FakeSched:
    def __init__(self, ac, head):
        self.config = type("C", (), {"num_train_timesteps": len(ac), "prediction_type": head, "clip_sample": True,
                                     "variance_type": "fixed_small"})()
        self.alphas_cumprod = torch.tensor(ac)
        ac = np.asarray(ac); prev = np.concatenate([[1.0], ac[:-1]])
        self.betas = torch.tensor(1 - ac / prev)


class _FakePolicy:
    def __init__(self, ac, head):
        self.noise_scheduler = _FakeSched(ac, head)
        self.model = lambda x, t, local_cond=None, global_cond=None: torch.zeros_like(x)
        self.conditional_sample = lambda *a, **k: None


def test_install_reports_grid_and_counts_nfe():
    ac = _cosine_alphas_cumprod()
    pol = _FakePolicy(ac, "sample")
    ts = S.install(pol, "ddim", 4)
    d = ts.describe()
    assert d["timesteps"] == [99, 74, 49, 24] and d["nfe_per_call"] == 4 and d["head"] == "sample"
    assert 0 < d["alpha_bar_first"] < 0.01  # t=99 of the cosine schedule is close to, not exactly, pure noise
    data = torch.zeros(2, 4, 3); mask = torch.zeros(2, 4, 3, dtype=torch.bool)
    out = pol.conditional_sample(data, mask, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 4, 3) and ts.nfe == 4 and ts.calls == 1
    with pytest.raises(ValueError):
        S.install(pol, "ddpm", 10)  # the anchor must run all T steps
    ts.uninstall()


def test_noise_hook_is_used_for_initial_noise():
    ac = _cosine_alphas_cumprod()
    pol = _FakePolicy(ac, "epsilon")
    ts = S.install(pol, "ddim", 1)
    seen = {}
    def noise_fn(shape, dtype, device, generator):
        seen["shape"] = shape
        return torch.full(shape, 0.5, dtype=dtype, device=device)
    ts.noise_fn = noise_fn
    data = torch.zeros(3, 4, 2); mask = torch.zeros(3, 4, 2, dtype=torch.bool)
    out = pol.conditional_sample(data, mask)
    assert seen["shape"] == (3, 4, 2)
    # eps head predicting zero noise: x0 = x_t / sqrt(a_99), then clipped to [-1, 1]
    assert torch.all(out == 1.0)


def test_eps_mode_raw_keeps_network_eps_for_eps_head_only():
    ac = _cosine_alphas_cumprod()
    x_t = torch.zeros(1, 2, 2, dtype=torch.float64); pred = torch.full((1, 2, 2), 5.0, dtype=torch.float64)  # clip binds
    x0r, eps_r = S.to_x0_eps(x_t, pred, ac[50], "epsilon", eps_mode="raw")
    x0c, eps_c = S.to_x0_eps(x_t, pred, ac[50], "epsilon", eps_mode="recompute")
    assert torch.allclose(x0r, x0c) and torch.allclose(eps_r, pred) and not torch.allclose(eps_c, pred)
    # x0 head: raw == recompute
    a, b = S.to_x0_eps(x_t, pred, ac[50], "sample", eps_mode="raw"); c, d = S.to_x0_eps(x_t, pred, ac[50], "sample")
    assert torch.allclose(a, c) and torch.allclose(b, d)
    with pytest.raises(ValueError):
        S.to_x0_eps(x_t, pred, ac[50], "epsilon", eps_mode="other")



def _betas_cosine(T=100, s=0.008):
    def f(t):
        return math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    return np.array([min(1 - f(i + 1) / f(i), 0.999) for i in range(T)])


def test_ddpm_step_matches_numpy_transcription_of_diffusers_0_11_1():
    betas = _betas_cosine(); ac = np.cumprod(1 - betas)
    rng = np.random.default_rng(3)
    for t in (99, 60, 7, 1, 0):
        x_t = rng.normal(size=(2, 5, 3)); eps = rng.normal(size=(2, 5, 3)) * 0.7; noise = rng.normal(size=(2, 5, 3))
        # numpy: diffusers 0.11.1 step(), prediction_type epsilon, clip_sample True, variance fixed_small
        a_t = ac[t]; a_prev = ac[t - 1] if t > 0 else 1.0; b_t = betas[t]
        x0 = np.clip((x_t - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t), -1, 1)
        c0 = math.sqrt(a_prev) * b_t / (1 - a_t); ct = math.sqrt(1 - b_t) * (1 - a_prev) / (1 - a_t)
        ref = c0 * x0 + ct * x_t
        if t > 0:
            ref = ref + math.sqrt(max((1 - a_prev) / (1 - a_t) * b_t, 1e-20)) * noise
        out = S.ddpm_step(torch.tensor(x_t), torch.tensor(eps), t, betas, ac, "epsilon", True, torch.tensor(noise))
        assert np.allclose(out.numpy(), ref, atol=1e-10), t
        # sample head with the equivalent x0 prediction gives the same x_prev
        x0_pred = (x_t - math.sqrt(1 - a_t) * eps) / math.sqrt(a_t)
        out2 = S.ddpm_step(torch.tensor(x_t), torch.tensor(x0_pred), t, betas, ac, "sample", True, torch.tensor(noise))
        assert np.allclose(out2.numpy(), ref, atol=1e-8), t


def test_sample_ddpm_counts_T_calls_and_uses_step_noise_hook():
    betas = _betas_cosine(); ac = np.cumprod(1 - betas)
    seen_t = []
    def net(x, t):
        seen_t.append(t); return torch.zeros_like(x)
    used = []
    def step_noise(t, like):
        used.append(t); return torch.zeros_like(like)
    x_T = torch.randn(1, 3, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    out, nfe = S.sample_ddpm(net, x_T, betas, ac, "epsilon", clip=True, step_noise_fn=step_noise)
    assert nfe == 100 and seen_t == list(range(99, -1, -1)) and used == list(range(99, 0, -1))
    assert torch.all(out.abs() <= 1.0 + 1e-12)


def test_conditioning_is_passed_through_to_the_network():
    ac = _cosine_alphas_cumprod()
    pol = _FakePolicy(ac, "epsilon")
    calls = []
    def model(x, t, local_cond=None, global_cond=None):
        calls.append((local_cond, global_cond)); return torch.zeros_like(x)
    pol.model = model
    ts = S.install(pol, "ddim", 2)
    lc = torch.ones(1, 4, 3); gc = torch.ones(1, 7)
    pol.conditional_sample(torch.zeros(1, 4, 2), torch.zeros(1, 4, 2, dtype=torch.bool), local_cond=lc, global_cond=gc)
    assert len(calls) == 2 and all(c[0] is lc and c[1] is gc for c in calls)


@pytest.mark.parametrize("k", [100, 10, 4, 1])
@pytest.mark.parametrize("use_clipped", [False, True])
def test_ddim_chain_matches_numpy_transcription_of_diffusers_0_11_1(k, use_clipped):
    """sample_ddim (eps head, trailing grid) against a numpy transcription of diffusers 0.11.1 ``DDIMScheduler.step``
    (eta=0, clip_sample=True, set_alpha_to_one=True; ``use_clipped_model_output`` False = raw / True = recompute) with
    ``prev_timestep = t - T // k`` -- which is exactly the trailing spacing, so the chains must agree for every k."""
    betas = _betas_cosine(); ac = np.cumprod(1 - betas); T = 100
    rng = np.random.default_rng(11)
    W = rng.normal(size=(6, 6)) * 0.3  # a fixed linear "network": eps = tanh(x @ W) * 0.8 + 0.05 * t/T
    def net_np(x, t):
        return np.tanh(x @ W) * 0.8 + 0.05 * t / T
    def net_torch(x, t):
        return torch.tensor(net_np(x.numpy(), t))
    x = rng.normal(size=(2, 6))
    ref = x.copy()
    for t in S.make_timesteps(T, k):
        prev_t = t - T // k
        a_t = ac[t]; a_prev = ac[prev_t] if prev_t >= 0 else 1.0; b_t = 1 - a_t
        eps = net_np(ref, t)
        x0 = (ref - math.sqrt(b_t) * eps) / math.sqrt(a_t)
        x0 = np.clip(x0, -1, 1)
        if use_clipped:
            eps = (ref - math.sqrt(a_t) * x0) / math.sqrt(b_t)
        ref = math.sqrt(a_prev) * x0 + math.sqrt(1 - a_prev) * eps
    out, nfe = S.sample_ddim(net_torch, torch.tensor(x), S.make_timesteps(T, k), ac, "epsilon", clip=True,
                             eps_mode="recompute" if use_clipped else "raw")
    assert nfe == k
    assert np.allclose(out.numpy(), ref, atol=1e-10)
