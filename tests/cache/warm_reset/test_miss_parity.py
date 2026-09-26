"""Per-bundle ``plain_k`` / ``full`` vs the process-level step_diag arm at B = 1 (CPU stub).

The reference is the step_diag plain arm: ``Pi05DiagInterceptor`` (no cache,
``exec_steps`` pinned on its stage-3 binding) serving one decision from a given
noise. The production side is the stack ``_wrap_policy`` builds for a
library-free ``miss`` bundle, direct and through the coordinator's MISS bucket
(one request). Actions must be bit-equal; real-model parity is
``test_miss_parity_manual.py``'s claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from exp.step_diag import pi05 as P
from exp.step_diag import recorder as R
from openpi.cache.timing import SystemTimer
from openpi.cache.warm_reset.types import private_noise
from tests.cache.test_interceptor import FakePolicy
from tests.cache.warm_reset._support import EPISODE, EXTRA, D, H, Pi05Model, obs_for
from tests.cache.warm_reset.test_miss_and_self_only import _miss_raw, _served


def _step_diag_plain(tmp_path, k: int, noise: np.ndarray):
    model = Pi05Model()
    mode = "full" if k == 10 else "plain"
    spec = R.DiagSpec(experiment_id="parity", env_id="pi05_libero_10", arm_id="full" if k == 10 else f"plain_k{k}",
                      mode=mode, k_full=10, action_shape=(H, D), exec_steps=k)
    icpt = P.Pi05DiagInterceptor(FakePolicy(model), timer=SystemTimer(enabled=False), eager=True,
                                 diag=R.DiagRecorder(spec, tmp_path / "diag"), mode=mode, exec_steps=k)
    icpt.on_task_begin()
    icpt.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    return icpt.infer(obs_for(None), noise=noise), model


@pytest.mark.parametrize("coordinator", [False, True])
@pytest.mark.parametrize("k", [1, 2, 3, 10])
def test_per_bundle_miss_equals_the_process_level_plain_arm(tmp_path, monkeypatch, coordinator, k):
    noise = private_noise(20260925 + k, (H, D)).numpy()
    ref, ref_model = _step_diag_plain(tmp_path / "ref", k, noise)
    stack = _served(tmp_path, monkeypatch, _miss_raw(tmp_path, k), coordinator=coordinator)
    stack.served.on_task_begin()
    stack.served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    out = stack.served.infer(obs_for(None), noise=noise)
    assert np.array_equal(out["actions"], ref["actions"])
    assert out["__hit_meta__"]["miss_nfe"] == k
    # both loops really ran k Euler steps, on the same timestep bit patterns
    assert len(stack.model.t_log) == len(ref_model.t_log) == k and stack.model.t_log == ref_model.t_log
