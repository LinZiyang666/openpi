"""Tests of the profile tools on the synthetic fixture (store + REAL harness runner output).

  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider exp/offline_search/profile/tests \
      --basetemp <scratch dir>
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from exp.offline_search.profile import breakdown as BD
from exp.offline_search.profile import common as C
from exp.offline_search.profile import compare as CMP
from exp.offline_search.profile import coverage as COV
from exp.offline_search.profile import explain as EX
from exp.offline_search.profile import fixture as FX
from exp.offline_search.profile import repr as RP
from exp.offline_search.profile import runprof as RUN
from exp.offline_search.profile import timeline as TL

hm = pytest.importorskip("exp.offline_search.harness.metrics")


@pytest.fixture(scope="session")
def fx(tmp_path_factory):
    base = tmp_path_factory.mktemp("fx")
    store = FX.build_store(base / "store")
    runs = FX.build_run(store, base / "run")
    COV.run(store, "auto", "all", procs=4, quiet=True)
    return {"store": store, "run": base / "run", "runs": runs, "out": base / "reports"}


# ------------------------------------------------------------------------------------------- metrics
def test_err_matrix_and_padding():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 16, 32))
    b = rng.normal(size=(9, 16, 32))
    sig = rng.uniform(0.2, 1.0, 7)
    E = C.err_matrix(C.scaled_seg(a, sig), C.scaled_seg(b, sig))
    ref = hm.err_seg(hm.seg(b)[None], hm.seg(a)[:, None], sig)
    assert np.abs(E - ref).max() < 1e-9
    b2 = b.copy()
    b2[..., 7:] = rng.normal(size=b2[..., 7:].shape) * 5      # padding dims (GR00T-like noise)
    b2[:, 5:, :] = 99.0                                       # non-executed steps
    assert np.array_equal(C.seg_err(b, a[0], sig), C.seg_err(b2, a[0], sig))
    rs = np.zeros((3, 32)); rs[:, :8] = 1; rs[:, 8:] = 7
    assert C.valid_rs(rs, "pi05").shape == (3, 8) and (C.valid_rs(rs, "pi05") == 1).all()


def test_aurc_semantics():
    rng = np.random.default_rng(1)
    e = rng.random(500)
    c = rng.integers(0, 4, 500).astype(float)
    assert C.aurc(c, e) == pytest.approx(hm.risk_coverage(e, c)["aurc"], abs=1e-15)
    assert C.aurc(np.ones(500), e, tie_aware=True) == pytest.approx(e.mean())
    assert C.aurc(-e, e) < C.aurc(e, e)                   # confident = low err is better
    ra = C.risk_at(c, e)
    rh = hm.risk_coverage(e, c)
    for k in (30, 50, 70, 90):
        assert ra[f"risk@{k}"] == pytest.approx(rh[f"risk_c{k}"])


def test_auroc_rows_and_spearman():
    rng = np.random.default_rng(2)
    S = rng.random((4, 30))
    G = rng.random((4, 30)) < 0.4
    assert np.allclose(C.auroc_rows(S, G), [C.auroc(S[i], G[i]) for i in range(4)])
    assert np.allclose(C.spearman_rows(S, 2 * S + 1), 1.0)


# ------------------------------------------------------------------------------------------ coverage
def test_coverage_bruteforce(fx):
    root = fx["store"]
    for cell in ("pi05_spatial_inf", "groot_l10_cache"):
        qc = C.QueryCell(root, cell)
        cur = C.open_library(root, qc.ms, "current")
        sig = C.lib_sigma(cur["action"])
        cc = C.load_coverage(root, "current")
        fl = C.load_floor(root, qc.ms, sig)
        for row in (0, 7, qc.n - 1):
            cand = cur.task_rows(qc.task_id[row])
            e = np.array([C.seg_err(cur["action"][r], qc["a_inf"][row], sig) for r in cand])
            j = int(np.argmin(e))
            assert cc.get(cell, "oracle_err")[row] == e[j]                     # bit-exact (harness arithmetic)
            assert cc.get(cell, "top_rows")[row, 0] == cand[j]
            thr = fl.thr(qc.third[row:row + 1])[0]
            assert cc.get(cell, "thr")[row] == pytest.approx(thr, rel=1e-6)
            assert cc.get(cell, "n_good")[row] == int((e <= thr).sum())
            assert cc.get(cell, "n_cand")[row] == cand.size
            rec = int(qc["rec_top1"][row])
            assert cc.get(cell, "rec_rank")[row] == int((e < e[list(cand).index(rec)]).sum()) + 1
            te = cc.get(cell, "top_errs")[row]
            k = min(50, cand.size)
            assert np.allclose(te[:k], np.sort(e)[:k], atol=1e-6) and np.isnan(te[k:]).all()


def test_coverage_equals_harness_oracle_and_reuses_cache(fx):
    root = fx["store"]
    cc = C.load_coverage(root, "current")
    for cell in C.CELLS:
        rc = C.RunCell(fx["runs"]["B0_current"], cell)
        assert np.array_equal(rc["oracle_err"], cc.get(cell, "oracle_err")[rc.rows])
        assert np.array_equal(rc["oracle_row"], cc.get(cell, "top_rows")[rc.rows, 0])
    r = COV.run(root, "current", "all", procs=2, quiet=True)
    assert r["current"]["meta"]["fingerprints"]      # cache kept, nothing recomputed
    meta, arrays = COV.read_cache(root, "current")
    assert sorted(arrays) == sorted(C.CELLS)


# ----------------------------------------------------------------------------------------- breakdown
def test_breakdown_matches_runner_json(fx):
    res = BD.run(fx["run"], None, fx["store"], quiet=True, out=str(fx["out"] / "bd"))
    for m, r in res.items():
        for row in r["rows"]:
            if row["slice"] != "all":
                continue
            j = json.loads((fx["run"] / m / f"{row['cell']}.json").read_text())["metrics"]
            for a, b in (("err", "err_mean"), ("indist", "indist"), ("aurc", "aurc"), ("regret", "regret_mean"),
                         ("oracle", "oracle_err_mean"), ("phase", "phase_err_mean"), ("risk@30", "risk_c30"),
                         ("risk@90", "risk_c90")):
                assert row[a] == pytest.approx(j[b], abs=1e-9), (m, row["cell"], a)
            assert row["grip"] == pytest.approx(j["grip_mis"], abs=1e-6)


def test_breakdown_slices_partition(fx):
    per = BD.cell_breakdown(fx["runs"]["fx_mixed"], "pi05_l10_cache", fx["store"])
    rows = per["rows"]
    n = rows[0]["n"]
    for name in ("task", "third", "outcome", "grip", "conf_dec", "cand", "good"):
        sl = [r for r in rows if r["slice"] == name]
        assert sl, name
        assert sum(r["n"] for r in sl) == n
        assert sum(r["n"] * r["err"] for r in sl) / n == pytest.approx(rows[0]["err"])
    assert any("mixed-library" in s for s in per["notes"])


# ------------------------------------------------------------------------------------------- compare
def test_compare_self_and_pair(fx):
    r = CMP.run(fx["runs"]["B0_current"], fx["runs"]["B0_current"], root=fx["store"], cells="pi05_spatial_inf",
                reps=200, procs=1, quiet=True, out=str(fx["out"] / "cmp_self"))["cells"][0]
    assert r["d_mean"]["est"] == 0 and r["d_mean"]["lo"] == 0 and r["d_mean"]["hi"] == 0
    assert r["flip"] == 0 and r["win"] == 0 and r["loss"] == 0 and r["d_aurc"]["est"] == 0
    r = CMP.run(fx["runs"]["B2_random"], fx["runs"]["B0_current"], root=fx["store"], cells="groot_l10_inf",
                reps=300, procs=1, quiet=True, out=str(fx["out"] / "cmp"))["cells"][0]
    a = C.RunCell(fx["runs"]["B2_random"], "groot_l10_inf")
    b = C.RunCell(fx["runs"]["B0_current"], "groot_l10_inf")
    assert r["d_mean"]["est"] == pytest.approx(a["err"].mean() - b["err"].mean())
    assert r["d_mean"]["lo"] <= r["d_mean"]["est"] <= r["d_mean"]["hi"]
    assert r["d_mean"]["est"] > 0                            # random is worse than B0
    assert r["aurc_a"] == pytest.approx(json.loads((fx["runs"]["B2_random"] / "groot_l10_inf.json").read_text())["metrics"]["aurc"])
    assert r["win"] + r["tie"] + r["loss"] == pytest.approx(1.0)


def test_aurc_bootstrap_identity():
    rng = np.random.default_rng(3)
    ep = np.repeat(np.arange(20), 7)
    e, c = rng.random(ep.size), rng.random(ep.size)
    bs = C.EpisodeBootstrap(ep, 5, 0)
    bs.W = np.ones_like(bs.W)                                # every episode once -> the plain AURC
    assert np.allclose(CMP._aurc_boot(c, e, bs), C.aurc(c, e))


# ------------------------------------------------------------------------------------------- explain
def test_explain_b0_fields_match_b0_extras(fx):
    md = fx["runs"]["B0_current"]
    res = EX.run(md, "pi05_l10_inf", root=fx["store"], worst=3, rand=3, quiet=True, out=str(fx["out"] / "ex"))
    rc = C.RunCell(md, "pi05_l10_inf")
    for d in res:
        p = d["pos"]
        t1 = d["b0"]["top1"]
        assert t1["row"] == rc["top1"][p]
        assert t1["cos_v0"] == pytest.approx(rc["x_cos_v0"][p], abs=2e-6)
        assert t1["cos_v1"] == pytest.approx(rc["x_cos_v1"][p], abs=2e-6)
        assert t1["rs_l2"] == pytest.approx(rc["x_dist_rs"][p], abs=2e-5)
        assert t1["fused"] == pytest.approx(rc["topk_scores"][p, 0], abs=2e-5)
        assert d["topk"][0]["err"] == pytest.approx(rc["err"][p])
        assert d["b0"]["oracle"]["err"] == pytest.approx(rc["oracle_err"][p])
        qc = C.QueryCell(fx["store"], "pi05_l10_inf")
        assert d["b0"]["b0_rec"]["row"] == qc["rec_top1"][d["row"]]  # the recorded online pick (fixture: key_v1 argmax)


def test_explain_mixed_library(fx):
    md = fx["runs"]["fx_mixed"]
    rc = C.RunCell(md, "groot_spatial_cache")
    odd = int(np.flatnonzero(rc["step"] % 2 == 1)[0])
    res = EX.run(md, "groot_spatial_cache", root=fx["store"], rows=[int(rc.rows[odd])], quiet=True, out=str(fx["out"] / "exm"))
    d = res[0]
    assert d["library"] == "bpool_all" and d["library_in_store"]
    assert d["topk"][0]["err"] == pytest.approx(rc["err"][odd])      # err computed on the bpool_all action
    assert d["oracle_rank_in_topk"] is None                         # oracle lives in library current


# ------------------------------------------------------------------------------------------ timeline
def test_moves_codes():
    traj = np.array([0, 0, 0, 0, 1, 1])
    step = np.array([0, 1, 2, 3, 0, 1])
    nxt = np.array([1, 2, 3, -1, 5, -1])
    sel = np.array([0, 1, 3, 3, 2, 4, 5])
    cont = np.array([False, True, True, True, True, True, False])
    code = TL.moves(sel, cont, traj, step, nxt)
    assert [TL.FLAG[c] for c in code] == [".", "T", "A", "=", "B", "S", "."]
    code2 = TL.moves(sel, cont, traj, step, None)             # without next: step + 1 on the same trajectory
    assert [TL.FLAG[c] for c in code2] == [".", "T", "A", "=", "B", "S", "."]


def test_timeline_runs(fx):
    r = TL.run(fx["run"], "fx_mixed", fx["store"], cells="pi05_spatial_inf", episodes=["0"], quiet=True,
               out=str(fx["out"] / "tl"))["fx_mixed"]
    s = r["summary"][0]
    assert s["switch"] + s["track"] + s["advance"] + s["stay"] + s["back"] == pytest.approx(1.0)
    ep = r["episodes"]["pi05_spatial_inf:0"]
    assert any(e["choice"].startswith("bpool_all/") for e in ep) and any(":" in e["oracle"] for e in ep)


# ---------------------------------------------------------------------------------------------- repr
def _oracle_encoder(view):
    """Test-only encoder: the executed action itself (library action / query a_inf) scaled by sigma."""
    sig = C.lib_sigma(view.lib.store["action"])
    a = view.get("action") if view.kind == "library" else view.get("a_inf")
    return C.scaled_seg(a, sig).astype(np.float32)


def test_repr_oracle_encoder_is_perfect(fx, monkeypatch):
    monkeypatch.setattr(RP, "user_encoder", lambda spec, metric, name=None: RP.Encoder("oracle_enc", _oracle_encoder, "l2"))
    r = RP.run(fx["store"], "pi05_l10,groot_spatial", "", encode="x:y", n_queries=40, procs=1, quiet=True,
               out=str(fx["out"] / "rp"))
    assert not r["errors"]
    for q in r["query"]:
        assert q["rho_mean"] == pytest.approx(1.0, abs=1e-6)
        assert q["orc_rank_p50"] == 1 and q["orc_top10"] == 1
        assert q["nn_err"] == pytest.approx(q["oracle_err"], abs=1e-5)
        assert q["auroc_top10"] == pytest.approx(1.0)


def test_repr_builtins(fx):
    r = RP.run(fx["store"], "pi05_spatial", "key_v0,rs", n_queries=30, procs=1, quiet=True, out=str(fx["out"] / "rpb"))
    lib = {x["repr"]: x for x in r["library"]}
    assert lib["rs"]["dim"] == 8 and lib["key_v0"]["dim"] == 48       # pi05 rs padding dropped
    assert 1.0 <= lib["key_v0"]["pr"] <= 48
    assert 0.0 <= lib["key_v0"]["w_sat99"] <= 1.0
    assert {q["cell"] for q in r["query"]} == {"pi05_spatial_inf", "pi05_spatial_cache"}


# ------------------------------------------------------------------------------------------- runprof
def test_runprof_summary_and_live(fx, capsys):
    r = RUN.summary(fx["run"], "B0_current", root=fx["store"], quiet=True, out=str(fx["out"] / "rp_sum"))["B0_current"]
    assert r["status"] == "DONE" and len(r["cells"]) == 8
    assert {s["section"] for s in r["sections"]} >= {"sim_v0", "sim_v1", "sim_rs", "fuse", "rank"}
    assert all(b["workers"] >= 1 and b["chunks"] >= 1 for b in r["balance"])
    RUN.live(fx["run"])
    assert "B0_current: DONE" in capsys.readouterr().out


def test_scaled_library(fx):
    from exp.offline_search.harness import store as hs
    base = hs.LibraryView(fx["store"], "pi05_l10", "current")
    s2 = RUN.ScaledLibrary(base, 2)
    assert s2.L == 2 * base.L
    nx = np.asarray(s2.next)
    ok = nx >= 0
    assert (s2.task_id[nx[ok]] == s2.task_id[np.flatnonzero(ok)]).all()
    assert (s2._copy[nx[ok]] == s2._copy[np.flatnonzero(ok)]).all()                  # next stays inside its copy
    assert (np.asarray(s2.episode)[nx[ok]] == np.asarray(s2.episode)[np.flatnonzero(ok)]).all()
    assert np.array_equal(s2.key_v0[base.L:], np.asarray(base.key_v0))
    h = RUN.ScaledLibrary(base, 0.5, seed=1)
    ep = np.asarray(h.episode)
    for e in np.unique(ep):                                                         # whole episodes kept
        assert (ep == e).sum() == (np.asarray(base.episode) == e).sum()
    assert set(h.tasks()) == set(base.tasks())


def test_scaling_runs(fx):
    r = RUN.scaling("exp.offline_search.harness.baselines:B0Current", {}, "pi05_spatial_inf", fx["store"], (0.5, 1, 2),
                    n_queries=30, quiet=True, out=str(fx["out"] / "scal"))
    Ls = [x["L"] for x in r["rows"]]
    assert Ls[1] == C.open_library(fx["store"], "pi05_spatial", "current").n and Ls[2] == 2 * Ls[1] and Ls[0] < Ls[1]
    assert all(x["ms_q"] > 0 for x in r["rows"]) and np.isfinite(r["slope"])


# ------------------------------------------------------------------------------ parallel == serial
def _strip(x):
    """Canonical JSON (NaN-safe) without report paths (they differ by --out) for equality checks."""
    def drop(v):
        if isinstance(v, dict):
            return {k: drop(w) for k, w in v.items() if k != "report_dir"}
        if isinstance(v, (list, tuple)):
            return [drop(w) for w in v]
        return v
    return json.dumps(drop(x), sort_keys=True, default=C._jsonable)


def test_parallel_outputs_identical(fx):
    o = fx["out"]
    st, run_ = fx["store"], fx["run"]
    a = BD.run(run_, None, st, quiet=True, out=str(o / "p1"), procs=1)
    b = BD.run(run_, None, st, quiet=True, out=str(o / "p4"), procs=4)
    assert _strip(a) == _strip(b)
    a = TL.run(run_, None, st, worst=1, quiet=True, out=str(o / "t1"), procs=1)
    b = TL.run(run_, None, st, worst=1, quiet=True, out=str(o / "t4"), procs=4)
    assert _strip(a) == _strip(b)
    a = CMP.run(fx["runs"]["fx_mixed"], fx["runs"]["B2_random"], root=st, reps=400, procs=1, quiet=True, out=str(o / "c1"))
    b = CMP.run(fx["runs"]["fx_mixed"], fx["runs"]["B2_random"], root=st, reps=400, procs=6, quiet=True, out=str(o / "c6"))
    assert _strip(a) == _strip(b)                                     # chunked bootstrap == single pass
    a = EX.run(fx["runs"]["B0_current"], "groot_l10_cache", root=st, worst=20, rand=10, quiet=True, out=str(o / "e1"), procs=1)
    b = EX.run(fx["runs"]["B0_current"], "groot_l10_cache", root=st, worst=20, rand=10, quiet=True, out=str(o / "e4"), procs=4)
    assert len(a) >= EX.PAR_MIN and _strip(a) == _strip(b)
    a = RUN.summary(run_, None, root=st, quiet=True, out=str(o / "r1"), procs=1)
    b = RUN.summary(run_, None, root=st, quiet=True, out=str(o / "r4"), procs=4)
    assert _strip(a) == _strip(b)
    meta, arr = COV.read_cache(st, "bpool_all")
    COV.run(st, "bpool_all", "all", procs=1, force=True, quiet=True, out=str(o / "v1"))
    _, arr1 = COV.read_cache(st, "bpool_all")
    COV.run(st, "bpool_all", "all", procs=5, force=True, quiet=True, out=str(o / "v5"))
    _, arr5 = COV.read_cache(st, "bpool_all")
    for c in arr1:
        for f in arr1[c]:
            assert np.array_equal(arr1[c][f], arr5[c][f], equal_nan=True), (c, f)
            assert np.array_equal(arr1[c][f], arr[c][f], equal_nan=True), (c, f)
