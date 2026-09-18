"""CPU tests for the keyed randomness of ``eval_dp_steps_v2`` (G2 R1-B7): every draw is a pure function of
``(sampling_seed, episode_id, decision_idx, kind, timestep)`` and therefore independent of ``n_envs``, of the chunk
layout, of other episodes and of retries; the DDPM posterior noise differs per timestep and per row; a full sampler
call reproduces one episode's action chunk bit-for-bit whatever batch it sits in."""

import numpy as np
import pytest
import torch

from exp.dp_nfe import dp_sampler as S
from exp.dp_nfe.eval_dp_steps_v2 import KeyedNoise, keyed_normal, noise_seed
from tests.dp_nfe.test_x0_sampler import _FakePolicy, _betas_cosine


def _episode_rows(keyed: KeyedNoise, n_chunks: int, n_rows: int, shape=(4, 2)):
    """{episode_id: [initial noise of decision 0, decision 1]} by walking the runner protocol."""
    out = {}
    for _ in range(n_chunks):
        keyed.on_reset()
        d0 = keyed.initial_noise((n_rows,) + shape, torch.float32, "cpu")
        d1 = keyed.initial_noise((n_rows,) + shape, torch.float32, "cpu")
        for r, eid in enumerate(keyed.episode_ids(n_rows)):
            out[eid] = (d0[r].clone(), d1[r].clone())
    return out


def test_seed_is_a_pure_function_of_the_key():
    a = noise_seed(0, 100003, 2, "init", -1); b = noise_seed(0, 100003, 2, "init", -1)
    assert a == b and 0 <= a < 2 ** 64
    assert len({noise_seed(0, 100003, 2, "init", -1), noise_seed(1, 100003, 2, "init", -1), noise_seed(0, 100004, 2, "init", -1),
                noise_seed(0, 100003, 3, "init", -1), noise_seed(0, 100003, 2, "ddpm", -1), noise_seed(0, 100003, 2, "ddpm", 5)}) == 6


def test_initial_noise_is_independent_of_n_envs_and_chunking():
    big = _episode_rows(KeyedNoise(0, 100000, 25), n_chunks=2, n_rows=25)     # 50 episodes, 2 chunks of 25
    small = _episode_rows(KeyedNoise(0, 100000, 10), n_chunks=5, n_rows=10)   # the same 50 episodes, 5 chunks of 10
    single = _episode_rows(KeyedNoise(0, 100000, 1), n_chunks=50, n_rows=1)   # one episode per chunk (retry of one id)
    assert set(big) == set(small) == set(single) == {100000 + i for i in range(50)}
    for eid in big:
        for d in (0, 1):
            assert torch.equal(big[eid][d], small[eid][d]) and torch.equal(big[eid][d], single[eid][d])
        assert not torch.equal(big[eid][0], big[eid][1])  # decisions differ
    # a different sampling seed changes everything
    other = _episode_rows(KeyedNoise(1, 100000, 25), n_chunks=1, n_rows=25)
    assert all(not torch.equal(other[e][0], big[e][0]) for e in other)


def test_step_noise_is_keyed_per_row_and_timestep():
    k = KeyedNoise(0, 100000, 3); k.on_reset()
    k.initial_noise((3, 4, 2), torch.float32, "cpu")  # decision 0 in flight
    like = torch.zeros(3, 4, 2)
    n99 = k.step_noise(99, like); n98 = k.step_noise(98, like)
    assert not torch.equal(n99, n98) and not torch.equal(n99[0], n99[1])
    assert torch.equal(n99[1], keyed_normal((4, 2), 0, 100001, 0, "ddpm", 99))
    # the same row in a batch of one gets the same posterior noise
    k1 = KeyedNoise(0, 100001, 1); k1.on_reset(); k1.initial_noise((1, 4, 2), torch.float32, "cpu")
    assert torch.equal(k1.step_noise(99, torch.zeros(1, 4, 2))[0], n99[1])


def test_use_before_reset_is_an_error():
    k = KeyedNoise(0, 100000, 4)
    with pytest.raises(RuntimeError):
        k.initial_noise((4, 2, 2), torch.float32, "cpu")


@pytest.mark.parametrize("sampler,k", [("ddim", 4), ("ddpm", 100)])
def test_sampler_output_of_an_episode_is_batch_invariant(sampler, k):
    """A per-row network (no cross-row mixing) + keyed hooks: episode 100002 gives the same chunk whether it is row 2 of
    a batch of 4 or the only row of a batch of 1 (single-episode retry)."""
    betas = _betas_cosine(); ac = np.cumprod(1 - betas)
    pol = _FakePolicy(ac, "epsilon")
    pol.noise_scheduler.betas = torch.tensor(betas); pol.noise_scheduler.variance_type = "fixed_small"
    W = torch.tensor(np.random.default_rng(5).normal(size=(2, 2)) * 0.3, dtype=torch.float32)
    pol.model = lambda x, t, local_cond=None, global_cond=None: torch.tanh(x @ W) + 0.01 * t
    def run(start_seed, n_rows):
        ts = S.install(pol, sampler, k)
        keyed = KeyedNoise(0, start_seed, n_rows); keyed.on_reset()
        ts.noise_fn = lambda shape, dtype, device, gen: keyed.initial_noise(shape, dtype, device)
        ts.step_noise_fn = keyed.step_noise
        out = pol.conditional_sample(torch.zeros(n_rows, 4, 2), torch.zeros(n_rows, 4, 2, dtype=torch.bool))
        ts.uninstall(); return out
    batch = run(100000, 4)          # rows = episodes 100000..100003
    alone = run(100002, 1)          # episode 100002 alone
    assert torch.equal(batch[2], alone[0])
    assert not torch.equal(batch[1], batch[2])
