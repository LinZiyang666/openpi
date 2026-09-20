"""CPU tests for exp/step_diag/analysis/analyze_shadow.py on rows produced by the real recorder with
stub sampling: admission (terminal finalize, contiguous indices, duplicates, low coverage,
accepted-terminal join), task tables, the correlation readings (constant input / no conclusion /
consistent) and the LOTO feasibility table."""

import json

import numpy as np
import pytest
import torch

from exp.step_diag import metrics as M
from exp.step_diag import recorder as R
from exp.step_diag.analysis import analyze_shadow as A

H, D = 50, 32


def _spec():
    return R.DiagSpec(experiment_id="exp", env_id="pi05_rc", arm_id="shadow", mode="shadow", k_full=10, k_set=(1, 2, 3, 5),
                      warm_ts=(0.1, 0.2, 0.3), n_primary=4, n_dense_extra=28, action_shape=(H, D), config_sha="abc")


def _weights(tmp_path):
    lib = torch.randn(30, H, D)
    mask = M.executed_mask(D, 3)
    w, sigma, _ = M.frozen_action_weights(lib, mask)
    path = tmp_path / "w.npz"
    np.savez(path, w=w.numpy(), sigma=sigma.numpy(), active_mask=mask.numpy(), degenerate=np.asarray([]),
             env_id=np.asarray("pi05_rc"))
    return str(path)


def _run_episodes(rec, tasks, n_decisions=6, offset=0.0, fail_every=None):
    uid = 0
    for task, n_eps in tasks:
        for e in range(n_eps):
            uid += 1
            rec.begin_episode(R.EpisodeIdentity.from_episode_start(
                experiment="robocasa365", task=task, episode_id=e,
                extra_metadata={"task_uid": f"{task}:{uid}", "attempt": 1, "task_id": 0, "orig_init_state_idx": e, "seed": 2000000 + e}))
            for i in range(n_decisions):
                def sample(z, k, _o=offset):
                    return z + (0.0 if k == 10 else _o / k)
                def bad(z, k):
                    raise RuntimeError("x")
                rec.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=1, hit_type="MISS", start_t=None,
                           schedule_id="pi05_v1", sample=(bad if (fail_every and i % fail_every == 0) else sample),
                           resume=lambda x, t: x, top1=lambda: (0.5, "e", {0.3: torch.zeros(H, D)}))
            rec.finalize_episode(e % 2 == 0)


def test_admission_metrics_and_table(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path / "rows")
    _run_episodes(rec, [("A", 9), ("B", 9)], offset=2.0)
    rows = A.read_rows([rec.rows_path])
    w, mask, _ = A.load_weights(_weights(tmp_path))
    res = A.admit_and_measure(rows, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.9, min_episodes=8)
    assert res["rejections"] == {} and len(res["episodes"]) == 18
    assert set(res["per_task"]) == {"A", "B"} and res["per_task"]["A"]["published"]
    assert res["per_task"]["A"]["d"][1] > res["per_task"]["A"]["d"][3] > 0
    assert res["per_task"]["A"]["sr_shadow"] == 5 / 9


def test_rejections_low_coverage_duplicates_gaps_and_journal(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path / "rows")
    _run_episodes(rec, [("A", 2)], n_decisions=4, fail_every=2)  # half the decisions error -> coverage 0.5
    rows = A.read_rows([rec.rows_path])
    w, mask, _ = A.load_weights(_weights(tmp_path))
    res = A.admit_and_measure(rows, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.9, min_episodes=8)
    assert res["rejections"]["episode_low_coverage"] == 2 and res["rejections"]["decision_error_row"] == 4
    # duplicate row and index gap
    dup = [dict(rows[0]), *rows]
    res2 = A.admit_and_measure(dup, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.0, min_episodes=1)
    assert res2["rejections"]["duplicate_rows"] == 1
    gap = [r for r in rows if not (r["task_uid"].endswith(":1") and r.get("decision_idx") == 1)]
    res3 = A.admit_and_measure(gap, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.0, min_episodes=1)
    assert res3["rejections"]["decision_index_gap"] == 1
    # journal accepted set excludes an episode
    res4 = A.admit_and_measure(rows, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.0, min_episodes=1,
                               journal_accepted={("A:1", 1)})
    assert res4["rejections"]["not_accepted_terminal"] == 1
    # sha mismatch
    bad = [dict(r, arrays_sha256="0" * 64) if r["status"] == "ok" else r for r in rows]
    res5 = A.admit_and_measure(bad, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.0, min_episodes=1)
    assert res5["rejections"]["arrays_missing_or_sha_mismatch"] == 4


def _per_task(ds):
    return {t: {"published": True, "d": {1: d}} for t, d in ds.items()}


def test_correlation_readings_and_loto():
    tasks = [f"t{i}" for i in range(10)]
    d = {t: float(i if i < 3 else 10 + i) for i, t in enumerate(tasks)}  # a clear gap between flat and cliff d
    g = {t: 0.05 * i for i, t in enumerate(tasks)}  # monotone: rho = 1; cliff (g >= 0.15) from t3 on
    c = A.correlation(_per_task(d), g, boot=200)
    assert c["rho"] == pytest.approx(1.0) and c["reading"] == "consistent_association"
    flat = {t: 0.01 for t in tasks}
    assert A.correlation(_per_task(d), flat, boot=50)["reading"] == "no_conclusion_constant_or_too_few"
    small = {t: 0.01 * i for i, t in enumerate(tasks)}  # range 0.09 < 0.15
    assert A.correlation(_per_task(d), small, boot=50)["reading"] == "no_conclusion_gap_range"
    anti = {t: 0.05 * (9 - i) for i, t in enumerate(tasks)}
    assert A.correlation(_per_task(d), anti, boot=50)["reading"] == "no_conclusion"
    lo = A.loto(_per_task(d), g)
    assert lo["hits"] + lo["missed_cliffs"] + lo["false_cliffs"] + lo["correct_flat"] == 10 and lo["missed_cliffs"] == 0


def test_markdown_and_gaps(tmp_path):
    agg = {"per_k": {"1": {"per_task": {"A": {"success_rate": 0.2}, "B": {"success_rate": 0.9}}}}}
    ref = {"tasks": {"A": {"sr": 0.7}, "B": 0.95}}
    (tmp_path / "agg.json").write_text(json.dumps(agg))
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    gaps = A.ladder_gaps(str(tmp_path / "agg.json"), str(tmp_path / "ref.json"))
    assert gaps["A"] == pytest.approx(0.5) and gaps["B"] == pytest.approx(0.05)
    res = {"n_episodes_seen": 1, "episodes": [], "rejections": {}, "per_task": {}}
    assert "shadow diagnostics" in A.markdown("pi05_rc", res, None, None)


def test_stratified_missing_table(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path / "rows")
    _run_episodes(rec, [("A", 4)], n_decisions=4, fail_every=2)  # every episode: 2 error rows of 4
    rows = A.read_rows([rec.rows_path])
    w, mask, _ = A.load_weights(_weights(tmp_path))
    res = A.admit_and_measure(rows, rec.rows_path.parent, w, mask, h_exec=5, n_primary=4, min_coverage=0.9, min_episodes=1)
    st = res["stratified"]
    assert st["success"]["episodes"] == 2 and st["failure"]["episodes"] == 2
    assert st["success"]["error_rate"] == 0.5 and st["success"]["rejected"] == 2  # coverage 0.5 < 0.9
    assert "missing labels by outcome" in A.markdown("pi05_rc", res, None, None)
