"""Unit tests for the R9 fable exploration tools (synthetic data only; no capture roots)."""
import numpy as np
import pandas as pd
import pytest

from exp.offline_search.rounds.r09.explore_fable.tools import common, paired, shadow_gap, task_alloc


def _matrix(seed=0, n_init=30, variants=("A", "CU", "P10"), p=None, cost=None, ndec=40):
    """Synthetic paired outcome matrix: task 0 hard for A, tasks 1-2 easy."""
    rng = np.random.default_rng(seed)
    p = p or {"A": [0.3, 0.95, 0.95], "CU": [0.8, 0.95, 0.95], "P10": [0.9, 0.95, 0.95]}
    cost = cost or {"A": 0.076, "CU": 0.30, "P10": 0.5}
    rows = []
    for t in range(3):
        for i in range(n_init):
            row = dict(task_id=t, init=i)
            for v in variants:
                row[f"{v}__success"] = float(rng.random() < p[v][t])
                row[f"{v}__n_dec"] = float(ndec)
                row[f"{v}__cost"] = float(cost[v] * ndec)
            rows.append(row)
    return pd.DataFrame(rows).set_index(["task_id", "init"]).sort_index()


def test_parse_arm_roundtrip():
    assert common.parse_arm(common.arm_name("pi05", "l10", 50, "CU")) == ("pi05", "l10", 50, "CU")
    assert common.parse_arm("r8_groot_spatial_P10") == ("groot", "spatial", None, "P10")
    with pytest.raises(ValueError):
        common.parse_arm("foo_bar")


def test_bootstrap_shapes_and_ci():
    vals = np.column_stack([np.ones(30), np.arange(30)])
    task = np.repeat([0, 1, 2], 10)
    boot = common.task_strat_bootstrap(vals, task, n_boot=50, seed=1)
    assert boot.shape == (50, 2)
    assert np.allclose(boot[:, 0], 1.0)
    lo, hi = common.ci(boot)
    assert lo[0] == hi[0] == 1.0 and lo[1] <= hi[1]


def test_mixture_and_summarize_match_pure_arms():
    M = _matrix()
    for v in ("A", "CU", "P10"):
        s = paired.summarize(M, pd.Series(v, index=M.index), n_boot=0)
        assert s["sr"] == pytest.approx(M[f"{v}__success"].mean())
        assert s["ir"] == pytest.approx(M[f"{v}__cost"].sum() / M[f"{v}__n_dec"].sum())
    assign = pd.Series(["CU" if t == 0 else "A" for t in M.index.get_level_values("task_id")], index=M.index)
    s = paired.summarize(M, assign, n_boot=100, reference="A")
    expect = (M["CU__success"][M.index.get_level_values("task_id") == 0].sum()
              + M["A__success"][M.index.get_level_values("task_id") != 0].sum()) / len(M)
    assert s["sr"] == pytest.approx(expect)
    assert s["ir"] == pytest.approx((0.30 + 2 * 0.076) / 3)
    assert s["sr_ci"][0] <= s["sr"] <= s["sr_ci"][1]
    assert s["wins"] >= 0 and s["losses"] >= 0


def test_mixture_rejects_unknown_variant():
    M = _matrix()
    with pytest.raises((KeyError, ValueError)):
        paired.mixture(M, pd.Series("ZZ", index=M.index))


def test_lagrangian_picks_cheap_when_lambda_large():
    M = _matrix()
    T = paired.per_task_table(M, ["A", "CU", "P10"])
    cheap = paired.lagrangian_task_allocation(T, ["A", "CU", "P10"], lam=1e9)
    assert set(cheap.values()) == {"A"}
    greedy = paired.lagrangian_task_allocation(T, ["A", "CU", "P10"], lam=0.0)
    assert greedy[0] == "P10"          # hard task: highest SR wins at lam=0
    # the hull is monotone in IR and SR
    pts = paired.hull(paired.task_frontier(M, ["A", "CU", "P10"], range(30), range(30), n_boot=0), "eval_sr", "eval_ir")
    irs = [p["eval_ir"] for p in pts]
    srs = [p["eval_sr"] for p in pts]
    assert irs == sorted(irs) and srs == sorted(srs)


def test_cv_frontier_uses_disjoint_folds():
    M = _matrix(n_init=30)
    folds = [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]
    pts = paired.cv_frontier(M, ["A", "CU"], folds, lams=[0.0, 1e9], n_boot=0)
    assert all(0 <= p["sr"] <= 1 for p in pts)
    assert any(abs(p["ir"] - 0.076) < 1e-9 for p in pts)   # lam=1e9 -> all A


def test_uniform_match_solves_fraction():
    M = _matrix()
    ref = task_alloc.uniform_match(M, 0.2, "A", "CU")
    assert ref is not None
    ir = ref["cost"].sum() / ref["n_dec"].sum()
    assert ir == pytest.approx(0.2, abs=1e-9)
    assert task_alloc.uniform_match(M, 0.9, "A", "CU") is None


def test_rank_allocation_top_k():
    sc = pd.Series({0: 0.9, 1: 0.1, 2: 0.5})
    assert task_alloc.rank_allocation(sc, 1, "CU", "A") == {0: "CU", 1: "A", 2: "A"}
    assert task_alloc.rank_allocation(sc, 2, "CU", "A") == {0: "CU", 1: "A", 2: "CU"}
    assert set(task_alloc.rank_allocation(sc, 0, "CU", "A").values()) == {"A"}


def test_gap_metrics_zero_and_scale():
    sigma = np.ones(7, np.float32)
    a = np.random.default_rng(0).standard_normal((5, 10, 7)).astype(np.float32)
    m, g, c = shadow_gap.gap_metrics(a, a, sigma)
    assert np.allclose(m, 0) and np.allclose(g, 0) and np.allclose(c, 0)
    b = a.copy()
    b[:, :5, :6] += 1.0
    m, g, _ = shadow_gap.gap_metrics(a, b, sigma)
    assert np.allclose(m, 1.0)
    b = a.copy()
    b[:, :5, 6] = -a[:, :5, 6]
    _, g, _ = shadow_gap.gap_metrics(a, b, sigma)
    assert np.allclose(g, 1.0)
    # sigma scaling halves the metric
    m2, _, _ = shadow_gap.gap_metrics(a, a + 1.0, 2 * sigma)
    assert np.allclose(m2, 0.5)


def test_auroc():
    assert shadow_gap.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert shadow_gap.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert shadow_gap.auroc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5
    assert np.isnan(shadow_gap.auroc([0.1, 0.2], [1, 1]))


def test_per_episode_summary_fields():
    table = pd.DataFrame(dict(
        episode_key=["e"] * 4, task_id=[1] * 4, init=[3] * 4, decision_seq=[0, 1, 2, 3], is_look=[True, False, True, False],
        gap_motion=[0.1, 0.2, 0.3, 0.4], gap_grip=[0, 0, 1, 1], x_d1=[1.0, np.nan, 3.0, np.nan], conf=[1, 2, 3, 4],
        x_dst=[0.5, 0.5, 0.5, 0.5], x_disp5=[0.1, 0.1, 0.1, 0.1], journal_success=[True] * 4))
    epi = shadow_gap.per_episode(table, ks=(1, 2))
    row = epi.iloc[0]
    assert row.gap_first1 == pytest.approx(0.1) and row.gap_first2 == pytest.approx(0.15)
    assert row.gap_all == pytest.approx(0.25) and row.gaplook_first1 == pytest.approx(0.1)
    assert row.d1_0 == 1.0 and row.d1_mean_looks == pytest.approx(2.0)


def test_kernel_synthesis_reproduces_weighted_mean():
    from exp.offline_search.rounds.r09.explore_fable.tools import synth_eval
    from exp.offline_search.rounds.r02.g1_awm.awm import _kernel_w

    class T:
        pass

    class M:
        kref = 5

    rng = np.random.default_rng(0)
    M.act = rng.standard_normal((40, 10, 32)).astype(np.float32)
    M.blind_rs = rng.standard_normal((40, 8)).astype(np.float32)
    rows = np.array([rng.permutation(40)[:16]])
    dist = np.sort(rng.random((1, 16)), axis=1)
    zq, zn = rng.standard_normal((1, 6)).astype(np.float32), rng.standard_normal((1, 16, 6)).astype(np.float32)
    out = synth_eval.synth_rules(M, rows, dist, zq, zn, M.blind_rs[rows[0]][:1], M.blind_rs[rows])
    w = _kernel_w(dist - dist[:, :1], 5)
    w = w / w.sum()
    expect = np.einsum("k,khd->hd", w[0], M.act[rows[0], :10, :7])
    assert np.allclose(out["kernel"][0], expect, atol=1e-6)
    assert np.allclose(out["uniform_1"][0], M.act[rows[0, 0], :10, :7])
    assert set(np.unique(out["vote_grip"][0][:, 6])) <= {-1.0, 1.0}
    assert out["llr_state"].shape == (1, 10, 7) and np.isfinite(out["llr_state"]).all()


def test_student_features_and_mlp_shapes():
    from exp.offline_search.rounds.r09.explore_fable.tools import student
    import torch
    n = 12
    D = dict(task=np.arange(n) % 10, st=np.zeros((n, 8), np.float32), k3=np.zeros((n, 64), np.float32),
             kw=np.zeros((n, 64), np.float32), served=np.zeros((n, 10, 7), np.float32))
    sigma = np.ones(7, np.float32)
    assert student.features(D, sigma, True, True).shape == (n, 64 + 64 + 10 + 8 + 70)
    assert student.features(D, sigma, False, False).shape == (n, 18)
    net = student.MLP(18, 70, width=16, depth=2)
    assert net(torch.zeros(3, 18)).shape == (3, 70)


def test_extend_method_keeps_frozen_rows_and_adds_codes():
    from exp.offline_search.rounds.r09.explore_fable.tools import grow_library

    class T:
        pass

    class M:
        pass

    rng = np.random.default_rng(1)
    L0, d = 12, 136
    m = M()
    m.act = rng.standard_normal((L0, 10, 32)).astype(np.float32)
    m.lib_ep = np.arange(L0, dtype=np.int32) // 4
    m.lib_step = np.arange(L0, dtype=np.int32) % 4
    m.blind_next = np.full(L0, -1, np.int32)
    m.blind_rs = rng.standard_normal((L0, 8)).astype(np.float32)
    m.blind_event = np.zeros(L0, bool)
    m.blind_terminal = np.zeros(L0, bool)
    m.sig = np.ones(7, np.float32)
    m.kref, m.k, m.H, m.cand_name, m.os_library, m.name = 5, 16, 10, "current", "current", "AWM"
    m.n_cand = {}
    t = T()
    t.rows = np.arange(L0, dtype=np.int64)
    t.Wf = np.eye(d, dtype=np.float32)
    t.shift = np.zeros(d, np.float32)
    t.Z = rng.standard_normal((L0, d)).astype(np.float32)
    t.z2 = (t.Z ** 2).sum(1)
    t.HD = (m.act[:, :5, :7]).reshape(L0, -1)
    t.h2 = (t.HD ** 2).sum(1)
    t.RS = m.blind_rs.copy()
    t.rs2 = (t.RS ** 2).sum(1)
    t.Vm0 = np.zeros(64, np.float32); t.V0 = np.zeros((L0, 64), np.float32)
    t.Vm1 = np.zeros(64, np.float32); t.V1 = np.zeros((L0, 64), np.float32)
    t.A0 = np.eye(d, dtype=np.float32); t.As0 = None; t.Z0 = None
    t.n20 = (t.Z ** 2).sum(1)
    m.tasks = {0: t}
    n = 5
    rows = dict(task=np.zeros(n, np.int64), init=np.zeros(n), seq=np.arange(n), P0=rng.standard_normal((n, 64)).astype(np.float32),
                P1=rng.standard_normal((n, 64)).astype(np.float32), rs=rng.standard_normal((n, 8)).astype(np.float32),
                chunk=rng.standard_normal((n, 10, 7)).astype(np.float32), ep=np.array(["e"] * n), arm=np.array(["a"] * n))
    g, l0 = grow_library.extend_method(m, rows, 10)
    assert l0 == L0 and g.act.shape == (L0 + n, 10, 32) and g.cand_name == "grown"
    assert np.array_equal(g.act[:L0], m.act) and np.allclose(g.act[L0:, :, :7], rows["chunk"]) and np.all(g.act[L0:, :, 7:] == 0)
    T0 = g.tasks[0]
    assert len(T0.rows) == L0 + n and np.array_equal(T0.rows, np.sort(T0.rows))
    x = np.concatenate([rows["P0"], rows["P1"], rows["rs"]], 1)
    assert np.allclose(T0.Z[L0:], x, atol=1e-6)            # identity metric: code == features
    assert np.allclose(T0.z2, (T0.Z.astype(np.float64) ** 2).sum(1), rtol=1e-4)
    assert T0.n20.shape == (L0 + n,)
    # the frozen method object is untouched
    assert m.act.shape[0] == L0 and len(m.tasks[0].rows) == L0
    # emulate_query returns one chunk per query, using only same-task rows
    pred = grow_library.emulate_query(g, np.zeros(2, np.int64), rows["P0"][:2], rows["P1"][:2], rows["rs"][:2])
    assert pred.shape == (2, 10, 7) and np.isfinite(pred).all()


def test_combo_method_validates_task_probabilities():
    from exp.offline_search.rounds.r09.explore_fable.tools import combo_method
    kw = dict(base_kwargs=dict(lib="current", kref=5, serving="anchor_tail", budget=1, gates="budget_only"))
    m = combo_method.ResidualTaskCalls([0, 0, 0.5, 0, 0, 0, 1, 0, 0, 0], "x.pkl", kw)
    assert m.name == "R9F_combo_00h0001000" and m.task_p[2] == 0.5 and m.p == 1.0
    with pytest.raises(ValueError):
        combo_method.ResidualTaskCalls([0] * 9, "x.pkl", kw)
    with pytest.raises(ValueError):
        combo_method.ResidualTaskCalls([2] + [0] * 9, "x.pkl", kw)
