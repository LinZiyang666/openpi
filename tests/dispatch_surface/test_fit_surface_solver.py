"""Tests for fit_surface: LP monotonicity, OOF-deployed-verdict parity,
mechanical delta rule, cohort audit, sparse-cell ladder, boundary export."""

from __future__ import annotations

import json

import numpy as np
import pytest

from exp.dispatch_surface.fit_surface import (
    GRID_LADDER_SV,
    N_FOLDS,
    FoldModel,
    Table,
    audit_cohort,
    choose_grid,
    equal_freq_edges,
    evaluate_candidate_deployed,
    export_boundaries,
    fit_bimonotone_quantile,
    order_statistic_offset,
    select_delta,
    min_cell_occupancy,
)
from openpi.cache.components.surface_judge import surface_verdict


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, n)
    v = rng.uniform(0, 1, n)
    y7 = 0.5 * (1 - s) + 0.3 * v + rng.exponential(0.05, n)
    y10 = y7 + 0.2 + rng.exponential(0.05, n)
    return s, v, y7, y10


def _bins(vals, edges):
    return np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)


def _fit(alpha=0.1, n_s=5, n_v=4):
    s, v, y7, y10 = _synthetic()
    s_edges = equal_freq_edges(s, n_s, 8)
    v_edges = equal_freq_edges(v, n_v, 8)
    q = fit_bimonotone_quantile(
        _bins(s, s_edges), _bins(v, v_edges), y7, y10,
        len(s_edges) - 1, len(v_edges) - 1, alpha,
    )
    return q, s_edges, v_edges


def _table(n=400, seed=0, split="fit"):
    s, v, y7, y10 = _synthetic(n, seed)
    task = np.arange(n) % 10
    init = (np.arange(n) // 10) % 5
    return Table(
        s=s, v=v, y7=y7, y10=y10,
        episode=np.array([f"ep_{t}_{i}" for t, i in zip(task, init)]),
        task=task, init_idx=init, split=np.array([split] * n),
    )


# ------------------------------------------------------------------
# LP / boundaries / order statistic
# ------------------------------------------------------------------


def test_lp_respects_all_three_monotonicities():
    q, _, _ = _fit()
    assert (np.diff(q, axis=1) <= 1e-8).all()
    assert (np.diff(q, axis=2) >= -1e-8).all()
    assert (q[0] <= q[1] + 1e-8).all()


def test_order_statistic_offset_matches_definition():
    res = list(np.arange(1, 51, dtype=float))
    assert order_statistic_offset(res, 0.05) == 49.0
    assert order_statistic_offset(res[:18], 0.05) == float("inf")


def test_export_boundaries_conservative_and_nested():
    q, s_edges, _ = _fit()
    delta = float(np.quantile(q[1], 0.5))
    full, warm = export_boundaries(q, s_edges, delta)
    assert (warm <= full).all()
    for j in range(q.shape[2]):
        ok = np.where(q[1, :, j] <= delta)[0]
        if ok.size:
            assert full[j] == s_edges[int(ok.min()) + 1]
        else:
            assert full[j] == np.inf


# ------------------------------------------------------------------
# Deployed-verdict delta evaluation (G2-B1 regression)
# ------------------------------------------------------------------


def _one_fold_models(table, fit_mask):
    q, s_edges, v_edges = _fit()
    return [FoldModel(q=q, s_edges=s_edges, v_edges=v_edges,
                      heldout_local=np.ones(int(fit_mask.sum()), dtype=bool))]


def test_low_s_rows_cannot_be_accepted():
    """Regression for G2-B1: the old proxy ignored s entirely. Rows whose s
    sits below every boundary must contribute ZERO hitshare no matter how
    small their quantile predictions are."""
    table = _table()
    table.s[:] = -1e9  # far below any exported boundary
    fit_mask = np.ones(len(table.s), dtype=bool)
    models = _one_fold_models(table, fit_mask)
    hitshare, acc = evaluate_candidate_deployed(
        1e9, models, table, fit_mask, 0.0, uses_disagreement=True,
    )
    assert hitshare == 0.0 and acc == 1.0


def test_deployed_evaluation_matches_surface_verdict_row_by_row():
    table = _table()
    fit_mask = np.ones(len(table.s), dtype=bool)
    models = _one_fold_models(table, fit_mask)
    m = models[0]
    delta = float(np.quantile(m.q[1], 0.6))
    full, warm = export_boundaries(m.q, m.s_edges, delta)
    hitshare, _ = evaluate_candidate_deployed(
        delta, models, table, fit_mask, 0.0, uses_disagreement=True,
    )
    manual = [
        surface_verdict(float(table.s[i]), float(table.v[i]), m.v_edges, full, warm,
                        uses_disagreement=True)
        for i in range(len(table.s))
    ]
    assert hitshare == pytest.approx(np.mean([x != "miss" for x in manual]))


def test_v_outside_fold_support_is_missed_in_evaluation():
    table = _table()
    fit_mask = np.ones(len(table.s), dtype=bool)
    models = _one_fold_models(table, fit_mask)
    table.v[:] = models[0].v_edges[-1] + 100.0  # beyond fitted support
    hitshare, _ = evaluate_candidate_deployed(
        1e9, models, table, fit_mask, 0.0, uses_disagreement=True,
    )
    assert hitshare == 0.0


def test_select_delta_rules():
    grid = np.array([0.1, 0.2, 0.3])
    metrics = {0.1: (0.45, 0.95), 0.2: (0.60, 0.93), 0.3: (0.60, 0.92)}
    assert select_delta(grid, metrics, 0.05) == (0.2, "qualified")
    metrics = {0.1: (0.10, 0.95), 0.2: (0.20, 0.95), 0.3: (0.0, 0.95)}
    assert select_delta(grid, metrics, 0.05) == (0.2, "fallback_accuracy_only")
    metrics = {d: (0.5, 0.10) for d in grid}
    assert select_delta(grid, metrics, 0.05)[0] is None


def test_no_pseudo_conformal_below_19_is_reachable():
    assert order_statistic_offset([1.0] * 10, 0.05) == float("inf")
    assert N_FOLDS == 5


def test_tau_start_t_mapping_pinned():
    assert round(1.0 - 7 / 10, 4) == 0.3
    assert round(1.0 - 10 / 10, 4) == 0.0


# ------------------------------------------------------------------
# 2-D sparse-cell ladder (G2-B8)
# ------------------------------------------------------------------


def test_min_cell_occupancy_counts_empty_cartesian_cells():
    sb = np.array([0] * 20 + [1] * 2)
    vb = np.array([0] * 20 + [1] * 2)
    # Cells (0,1) and (1,0) are EMPTY: the minimum over ALL cartesian cells
    # is 0, not the min over occupied ones (G2R2-B3).
    assert min_cell_occupancy(sb, vb, 2, 2) == 0


def test_choose_grid_rejects_correlated_empty_cells():
    """Regression for G2R2-B3: s=v leaves 83% of joint cells EMPTY at 12x6.
    The old non-empty-only rule reported sparse_fraction=0 and accepted; the
    every-cartesian-cell rule must refuse every rung and stop-loss."""
    x = np.linspace(0, 1, 1200)
    assert choose_grid(x, x, GRID_LADDER_SV) is None


def test_choose_grid_accepts_dense_and_descends():
    rng = np.random.default_rng(0)
    dense_s, dense_v = rng.uniform(0, 1, 5000), rng.uniform(0, 1, 5000)
    assert choose_grid(dense_s, dense_v, GRID_LADDER_SV) is not None
    # 500 independent uniform samples: every 12x6 joint cell holds ~7 (<8),
    # so the top rung is refused and the ladder lands on a coarser one.
    s, v = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
    assert choose_grid(s, v, ((12, 6),)) is None
    edges = choose_grid(s, v, GRID_LADDER_SV)
    assert edges is not None
    assert (len(edges[0]) - 1, len(edges[1]) - 1) != (12, 6)


# ------------------------------------------------------------------
# Cohort audit (G2-B8)
# ------------------------------------------------------------------


def _manifest(tmp_path, files):
    p = tmp_path / "cohort.json"
    p.write_text(json.dumps({"files": files}))
    return str(p)


def _full_cohort_files():
    files = []
    for t in range(10):
        for j in range(5):
            files.append({"task_id": t, "init_idx": j, "split": "fit"})
        for j in range(5, 15):
            files.append({"task_id": t, "init_idx": j, "split": "cal"})
    return files


def _cohort_table():
    rows_task, rows_init, rows_split, rows_ep = [], [], [], []
    for f in _full_cohort_files():
        for _ in range(2):  # two steps per episode
            rows_task.append(f["task_id"])
            rows_init.append(f["init_idx"])
            rows_split.append(f["split"])
            rows_ep.append(f"ep_{f['task_id']}_{f['init_idx']}")
    n = len(rows_task)
    return Table(
        s=np.zeros(n), v=np.zeros(n), y7=np.zeros(n), y10=np.zeros(n),
        episode=np.array(rows_ep), task=np.array(rows_task),
        init_idx=np.array(rows_init), split=np.array(rows_split),
    )


def test_audit_cohort_accepts_exact_quota(tmp_path):
    audit_cohort(_cohort_table(), _manifest(tmp_path, _full_cohort_files()),
                 verify_files=False)


def test_audit_cohort_rejects_repeated_identities_under_fresh_episode_ids(tmp_path):
    """G2R2-B2 adversarial reproduction: a table repeating 20 identities with
    a fresh episode_id per repeat used to pass. Coverage (150/150) and the
    identity<->episode bijection must both refuse it."""
    files = _full_cohort_files()
    table = _cohort_table()
    # Rewrite the table so only the first 20 identities appear, cycled, each
    # occurrence under a brand-new episode name.
    keys = [(f["task_id"], f["init_idx"], f["split"]) for f in files[:20]]
    for i in range(len(table.s)):
        t, init, split = keys[i % 20]
        table.task[i], table.init_idx[i] = t, init
        table.split[i] = split
        table.episode[i] = f"fresh_ep_{i}"
    with pytest.raises(SystemExit):
        audit_cohort(table, _manifest(tmp_path, files), verify_files=False)


def test_audit_cohort_rejects_episode_identity_bijection_break(tmp_path):
    files = _full_cohort_files()
    table = _cohort_table()
    # Same identity under two different episode ids.
    dup = np.where((table.task == table.task[0]) & (table.init_idx == table.init_idx[0]))[0]
    table.episode[dup[-1]] = "a_second_episode_for_same_identity"
    with pytest.raises(SystemExit):
        audit_cohort(table, _manifest(tmp_path, files), verify_files=False)


def test_audit_cohort_verifies_file_content(tmp_path):
    import hashlib

    payload = tmp_path / "ep.h5"
    payload.write_bytes(b"original")
    files = [{"task_id": 0, "init_idx": 0, "split": "fit", "path": str(payload),
              "sha256": hashlib.sha256(b"original").hexdigest()}]
    table = None  # file check fires before any table access
    payload.write_bytes(b"tampered")
    with pytest.raises(SystemExit):
        audit_cohort(table, _manifest(tmp_path, files), verify_files=True)


@pytest.mark.parametrize("corrupt", ["short_quota", "row_not_in_manifest", "label_flip"])
def test_audit_cohort_rejects(tmp_path, corrupt):
    files = _full_cohort_files()
    table = _cohort_table()
    if corrupt == "short_quota":
        files = files[:-1]
    elif corrupt == "row_not_in_manifest":
        table.init_idx[0] = 49
    else:
        table.split[0] = "cal"
    with pytest.raises(SystemExit):
        audit_cohort(table, _manifest(tmp_path, files), verify_files=False)
