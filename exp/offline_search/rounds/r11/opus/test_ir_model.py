"""Self-tests for the R11 opus IR model (CPU, seconds, no data outside out/anchor).

  PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m exp.offline_search.rounds.r11.opus.test_ir_model
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from exp.offline_search.rounds.r04.k1_blind.judge import BlindMixedJudge
from exp.offline_search.rounds.r11.opus import ir_model as M


def test_guard_port_matches_frozen_r4_progress(n_seq=300, seed=0):
    """ir_model.guard_flags == frozen BlindMixedJudge._progress span rule (span >= noprog_n - 1), anchors 2 apart."""
    rng = np.random.default_rng(seed)
    n_rows = 400
    C = SimpleNamespace(prog=rng.random(n_rows), ep_len=rng.integers(10, 105, n_rows))
    for _ in range(n_seq):
        L = int(rng.integers(2, 40))
        steps = np.arange(0, 2 * L, 2)
        # progress sequences with plateaus so both branches are exercised
        top1 = rng.integers(0, n_rows, L)
        if rng.random() < .5:
            top1[rng.random(L) < .4] = top1[0]
        fake = SimpleNamespace(_vision_progress=[], C=C, memo_reset_after_miss=False, prog_eps=.5)
        ref = []
        for s, t in zip(steps, top1):
            span = BlindMixedJudge._progress(fake, SimpleNamespace(step=int(s), prev_hit=True), int(t))
            ref.append(span >= 2)
        got = M.guard_flags(steps, C.prog[top1], C.ep_len[top1])
        assert np.array_equal(np.array(ref), got), (steps, top1)


def test_random_closed_form_matches_simulation():
    eps = M.load_episodes("pi05", "l10", 50)
    b = M.base_stats("pi05", eps)
    for rho in (.1, .4, .7):
        sim = M.simulate("pi05", eps, M.Random(rho), reps=16)["IR"]
        cf = M.random_closed_form("pi05", b["v"], b["g"], rho)
        assert abs(sim - cf) < .003, (rho, sim, cf)


def test_gapcap_bounds_cache_runs():
    eps = M.load_episodes("groot", "spatial", 50)
    for K in (0.0, 1.0, 1.4, 2.6):
        knob = M.DitheredGapCap(K)
        for ep in eps:
            g, k = knob.calls(ep, 0)
            assert not (g & k).any()
            run = mx = 0
            for c in (g | k):
                run = 0 if c else run + 1
                mx = max(mx, run)
            assert mx <= int(np.ceil(K)), (K, mx)


def test_knob_never_on_guard_anchor_and_ir_bounds():
    eps = M.load_episodes("pi05", "spatial", 50)
    for knob in (M.Random(.5), M.Random(.3, tail=2), M.DitheredGapCap(1.5, post_guard_tail=1), M.SigmaDelta(.5)):
        for ep in eps[:20]:
            g, k = knob.calls(ep, 1)
            assert not (g & k).any()
        r = M.simulate("pi05", eps, knob)
        assert M.base_stats("pi05", eps)["IR"] - 1e-9 <= r["IR"] <= .52


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
