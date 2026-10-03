"""Small CPU tests for the R10 offline corrector analysis (library data only; < 1 min).

Run: PYTHONPATH=.:src .venv/bin/python -m pytest -q exp/offline_search/rounds/r10/analysis_opus/tools/test_core.py
"""
import numpy as np
import pytest

from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r10.analysis_opus.tools.run_offline import build_pairs, pair_weights


def _toy(n_ep=5, rows=30, d=12, seed=0):
    rng = np.random.default_rng(seed)
    ep = np.repeat(np.arange(n_ep), rows)
    step = np.tile(np.arange(rows), n_ep)
    X = rng.standard_normal((len(ep), d)) + step[:, None] * 0.05
    act = rng.standard_normal((len(ep), 10, 7)).astype(np.float32)
    heads = act[:, :5, :7].reshape(len(ep), -1)
    return X, act, heads, ep, step


def test_loeo_never_retrieves_own_episode():
    X, act, heads, ep, step = _toy()
    met = core.TaskMetric(X, heads, ep, step)
    _, _, _, mem, w = core.serve(X, step, ep, met, X, None, ep, act, kref=5)
    assert (mem >= 0).all()
    assert not (ep[mem] == ep[:, None]).any()
    assert np.allclose(w.sum(1), 1)


def test_cand_mask_restricts_candidates():
    X, act, heads, ep, step = _toy()
    met = core.TaskMetric(X, heads, ep, step)
    keep = np.isin(ep, [1, 2])
    _, _, _, mem, _ = core.serve(X[ep == 0], step[ep == 0], ep[ep == 0], met, X, None, ep, act, kref=5, cand_mask=keep)
    assert np.isin(ep[mem], [1, 2]).all()


def test_topk_kernel_matches_awm_query_arithmetic():
    rng = np.random.default_rng(1)
    for kref in (5, 8):
        D = rng.random((7, 200))
        D[:, 3] = D[:, 4]                         # an exact tie: AWM orders ties by row index
        idx, Dk, w = core.topk_kernel(D, kref)
        for r in range(len(D)):
            dt = D[r]
            ref = np.argpartition(dt, 15)[:16]
            ref = ref[np.lexsort((ref, dt[ref]))]
            assert np.array_equal(idx[r], ref)
            wr = _kernel_w(dt[ref] - dt[ref][0], kref)
            assert np.allclose(w[r], wr / wr.sum())


def _fable_fit_head(X, Y, weights, seed=0, n_rff=384, alpha=100.0):
    """Verbatim arithmetic of explore_fable/round2/tools/corrector.py:fit_head (reference)."""
    rng = np.random.default_rng(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xn = np.clip((X - mean) / std, -8, 8)
    W = (rng.standard_normal((X.shape[1], n_rff)) / np.sqrt(X.shape[1])).astype(np.float32)
    b = rng.uniform(0, 2 * np.pi, n_rff).astype(np.float32)
    F = np.concatenate([Xn, np.cos(Xn @ W + b) * np.sqrt(2)], 1).astype(np.float64)
    sw = np.sqrt(weights / weights.mean())[:, None]
    Fw, Yw = F * sw, Y.astype(np.float64) * sw
    mu_f, mu_y = (Fw * sw).sum(0) / (sw ** 2).sum(), (Yw * sw).sum(0) / (sw ** 2).sum()
    Fc, Yc = Fw - mu_f * sw, Yw - mu_y * sw
    coef = np.linalg.solve(Fc.T @ Fc + alpha * np.eye(F.shape[1]), Fc.T @ Yc).T
    return dict(mean=mean, std=std, w=W, bias=b, coef=coef, intercept=mu_y - coef @ mu_f)


def test_chunked_head_equals_fable_fit_head():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((900, 40))
    Y = rng.standard_normal((900, 60))
    wts = rng.uniform(0.2, 2.0, 900)
    ref = _fable_fit_head(X, Y, wts)
    h = core.Head(X, Y, wts, chunk=128)                  # total=None == fable's normalization (sum w = n)
    Xt = rng.standard_normal((50, 40))
    Xn = np.clip((Xt - ref["mean"]) / ref["std"], -8, 8)
    Fr = np.concatenate([Xn, np.cos(Xn @ ref["w"] + ref["bias"]) * np.sqrt(2)], 1)
    assert np.allclose(h.predict(Xt), Fr @ ref["coef"].T + ref["intercept"], atol=1e-4)
    # total = n_rows / n_pairs rescales the effective ridge strength (alpha relative to sum of weights)
    h2 = core.Head(X, Y, wts, total=90)
    assert not np.allclose(h2.predict(Xt), h.predict(Xt), atol=1e-3)


def test_pairs_exclude_own_episode_respect_radius_and_cap_and_weights():
    X, act, heads, ep, step = _toy()
    met = core.TaskMetric(X, heads, ep, step)
    Z = met.code(X)
    D = core.pdist(Z, Z)
    D[ep[:, None] == ep[None, :]] = np.inf
    r = float(np.median(np.sort(D, 1)[:, 15]))
    a, b = build_pairs(D, ep, r, 8)
    assert (ep[a] != ep[b]).all() and (D[a, b] <= r).all() and np.bincount(a).max() <= 8
    w, n_anchor = pair_weights(a, ep)
    per_ep = np.bincount(ep[a], weights=w)
    assert np.allclose(per_ep[per_ep > 0], 1.0)           # equal weight per episode
    a2, b2 = build_pairs(D, ep, None, None, rng=np.random.default_rng(0), random_m=10)
    assert (ep[a2] != ep[b2]).all() and np.bincount(a2).min() == 10


def test_forbidden_paths_refused():
    with pytest.raises(ValueError):
        core.assert_library_path("/home/weiland/trace_runs/os_closed_loop/r08_main/arms.json")
    with pytest.raises(ValueError):
        core.assert_library_path("/home/weiland/trace_runs/offline_search_store/derived/r08/x.parquet")
    core.assert_library_path(core.STORE / "library" / "pi05_l10" / "bpool_cs" / "action.npy")


def test_subsets_nested_and_match_sol_rule():
    C = core.load_cell("pi05", "l10")
    prev = set()
    for size in (50, 100, 200, 500):
        rows, fold = core.subset_and_folds(C, size)
        assert prev <= set(rows.tolist())
        prev = set(rows.tolist())
        eps = np.unique(C["ep"][rows])
        assert len(eps) == size and all((C["task"][C["ep"] == e] == C["task"][C["ep"] == e][0]).all() for e in eps[:3])
        # every fold holds out whole episodes
        for e in eps[:20]:
            assert len(np.unique(fold[C["ep"][rows] == e])) == 1


def test_regime1_score_matches_awm_fresh_formula():
    from exp.offline_search.rounds.r10.analysis_opus.tools import run_regime as rr
    X, act, heads, ep, step = _toy()
    met = core.TaskMetric(X, heads, ep, step)
    sig = np.ones(7)
    H = rr.heads35(act, sig)
    pt = rr.prev_tail_of(np.arange(len(ep)), ep, step, act, sig)
    q = np.flatnonzero((ep == 0) & (step > 0))[:5]
    S, D = rr.scores(met, X[q], step[q], ep[q], pt[q], X, ep, H, 0.7, 1)
    for k, i in enumerate(q):
        adm = ep != ep[i]
        d = np.linalg.norm(met.code(X[adm]) - met.code(X[i:i + 1]), axis=1)
        tail = act[i - 1, 5:10, :7].ravel()
        c = np.sqrt(((act[adm, :5, :7].reshape(adm.sum(), -1) - tail) ** 2).mean(1))
        assert np.allclose(S[k, adm], d / np.median(d) + 0.5 * c / 0.7, atol=1e-6)
        assert np.isinf(S[k, ~adm]).all()
