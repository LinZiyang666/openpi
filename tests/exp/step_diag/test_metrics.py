"""CPU tests for exp/step_diag/metrics.py: the frozen deviation kernel over executed dims and the
executed window, the paired / dispersion / ratio quantities and their edge cases, and the mixture
fit's null behaviour on degenerate input."""

import numpy as np
import pytest
import torch

from exp.step_diag import metrics as M

H, D, N_EXEC, H_EXEC = 10, 8, 3, 5


def _weights():
    lib = torch.randn(20, H, D)
    lib[..., 2] = 0.5  # a constant executed dim
    lib[..., 5] = 0.0  # a constant padding dim
    mask = M.executed_mask(D, N_EXEC)
    w, sigma, degenerate = M.frozen_action_weights(lib, mask)
    return w, mask, degenerate


def test_frozen_weights_keep_constant_executed_dims_and_zero_padding():
    w, mask, degenerate = _weights()
    assert degenerate == [2]
    assert w[2] == 1.0 and (w[:N_EXEC] > 0).all() and (w[N_EXEC:] == 0).all()
    assert mask.tolist() == [True] * N_EXEC + [False] * (D - N_EXEC)


def test_deviation_ignores_padding_dims_and_steps_outside_the_window():
    w, mask, _ = _weights()
    a = torch.zeros(H, D)
    b = a.clone()
    b[:, N_EXEC:] += 5.0  # padding only
    b[H_EXEC:, :N_EXEC] += 5.0  # executed dims, but past the executed window
    assert M.deviation(a, b, w, mask, H_EXEC) == 0.0
    c = a.clone()
    c[0, 0] = 1.0 / w[0]
    assert M.deviation(a, c, w, mask, H_EXEC) == pytest.approx(1.0 / H_EXEC)


def test_per_timestep_mean_is_not_flattened_l2():
    w, mask, _ = _weights()
    w = torch.where(mask, torch.ones_like(w), torch.zeros_like(w))
    a = torch.zeros(H, D)
    b = a.clone()
    b[0, 0] = 3.0
    b[1, 1] = 4.0
    per_step_mean = (3.0 + 4.0) / H_EXEC
    flattened = 5.0
    assert M.deviation(a, b, w, mask, H_EXEC) == pytest.approx(per_step_mean)
    assert per_step_mean != pytest.approx(flattened)


def test_paired_dispersion_ratio_and_null_rules():
    w, mask, _ = _weights()
    full = [torch.randn(H, D) for _ in range(4)]
    assert M.paired_deviation(full, full, w, mask, H_EXEC) == 0.0  # k = K
    disp = M.dispersion(full, w, mask, H_EXEC)
    assert disp > 0
    assert M.ratio(0.0, disp) == 0.0
    assert M.ratio(1.0, 0.0) is None and M.ratio(1.0, float("nan")) is None
    same = [full[0]] * 4
    assert M.dispersion(same, w, mask, H_EXEC) == 0.0  # zero spread, ratio undefined
    with pytest.raises(ValueError):
        M.dispersion(full[:1], w, mask, H_EXEC)
    with pytest.raises(ValueError):
        M.paired_deviation(full[:3], full, w, mask, H_EXEC)


def test_same_noise_error_is_not_bounded_by_spread():
    """A reduced loop can move every sample by the same offset: zero spread change, positive d_k."""
    w, mask, _ = _weights()
    full = [torch.randn(H, D) for _ in range(4)]
    shifted = [x + 1.0 for x in full]
    assert M.paired_deviation(shifted, full, w, mask, H_EXEC) > 0
    assert M.dispersion(shifted, w, mask, H_EXEC) == pytest.approx(M.dispersion(full, w, mask, H_EXEC))


def test_decision_metrics_structure_and_rejections():
    w, mask, _ = _weights()
    full = [torch.randn(H, D) for _ in range(6)]
    a_k = {1: [x + 0.1 for x in full[:4]], 3: full[:4]}
    out = M.decision_metrics(full, a_k, {0.3: full[0], 0.5: None}, w, mask, H_EXEC, n_primary=4)
    assert out["n_primary"] == 4 and out["d"][3] == 0.0 and out["d"][1] > 0
    assert out["excess_vs_spread"][3] == pytest.approx(-out["disp_K"])
    assert out["d_w"][0.5] is None and out["d_w"][0.3] >= 0
    with pytest.raises(ValueError):
        M.decision_metrics(full[:3], a_k, {}, w, mask, H_EXEC, n_primary=4)
    with pytest.raises(ValueError):
        M.decision_metrics(full, {1: full[:2]}, {}, w, mask, H_EXEC, n_primary=4)


def test_mixture_fit_null_on_constant_input_and_dict_on_normal_input():
    w, mask, _ = _weights()
    const = [torch.zeros(H, D)] * 32
    res = M.mixture_fit(const, w, mask, H_EXEC)
    assert res["n_dims"] == N_EXEC and res["delta_bic_2_vs_1"] is None
    rng = np.random.default_rng(0)
    samples = [torch.as_tensor(rng.normal(size=(H, D)), dtype=torch.float32) for _ in range(32)]
    res2 = M.mixture_fit(samples, w, mask, H_EXEC)
    assert set(res2) >= {"delta_bic_2_vs_1", "pca_explained_2d", "n_samples", "n_dims"}
