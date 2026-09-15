"""Regression coverage for run provenance, restart evidence and fit input identity."""

import importlib
import json
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from exp.ablation_study.cache_size.run_size_eval import assert_accepted_full_hit, write_census
from exp.common.conductor_evidence import merge_hits, merge_source


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _terminal(**changes):
    return dict(yaml_id="arm", task_uid="arm:eval:0:0", phase="eval", status="done",
                success=True, accepted=True, run_id="current", attempt=1, **changes)


def _step(**changes):
    row = dict(yaml_id="arm", task_uid="arm:eval:0:0", step_idx=0, hit_type="FULL_HIT",
               accepted=True, run_id="current", attempt=1)
    row.update(changes)
    return row


@pytest.mark.parametrize("changed", [{"error": "CUDA failure"}, {"status": "failed"}, {"ts": 4.0}])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_terminal_duplicate_is_never_selected_by_order(tmp_path, changed, reverse):
    first = _terminal()
    rows = [first, {**first, **changed}]
    path = tmp_path / "journal.jsonl"
    _write_rows(path, rows[::-1] if reverse else rows)
    with pytest.raises(SystemExit, match="disagree"):
        merge_source({}, "test", path, {"arm"}, restrict=True)


def test_attempts_cannot_rank_two_driver_runs(tmp_path):
    path = tmp_path / "journal.jsonl"
    row = _terminal()
    _write_rows(path, [{**row, "attempt": 8}, {**row, "run_id": "restart", "attempt": 1}])
    with pytest.raises(SystemExit, match="run identities disagree"):
        merge_source({}, "test", path, {"arm"}, restrict=True)


def test_identical_terminal_copies_are_allowed(tmp_path):
    path = tmp_path / "journal.jsonl"
    _write_rows(path, [_terminal(), _terminal()])
    winners = {}
    merge_source(winners, "test", path, {"arm"}, restrict=True)
    merge_source(winners, "test", path, {"arm"}, restrict=True)
    assert len(winners["arm"]) == 1


def test_unselected_reference_records_are_excluded_before_validation(tmp_path):
    path = tmp_path / "journal.jsonl"
    _write_rows(path, [_terminal(), {"yaml_id": "other", "status": "done"}])
    winners = {}
    merge_source(winners, "reference", path, {"arm"}, restrict=False)
    assert set(winners) == {"arm"}


def test_step_duplicate_compares_payload_as_well_as_hit(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write_rows(a, [_step(entry_id="entry-A")])
    _write_rows(b, [_step(entry_id="entry-B")])
    with pytest.raises(SystemExit, match="different traces"):
        merge_hits([str(a), str(b)])


@pytest.mark.parametrize("old_only", [False, True])
def test_driver_exit_gate_uses_the_accepted_run_after_restart(tmp_path, old_only):
    journal, steps = tmp_path / "journal.jsonl", tmp_path / "per_step.jsonl"
    _write_rows(journal, [_terminal()])
    old = _step(run_id="before-crash", hit_type="FULL_HIT" if old_only else "MISS")
    _write_rows(steps, [old] if old_only else [old, _step()])
    if old_only:
        with pytest.raises(SystemExit, match="no inference rows"):
            assert_accepted_full_hit(journal, steps, {"arm"}, require_run_id=True)
    else:
        result = assert_accepted_full_hit(journal, steps, {"arm"}, require_run_id=True)
        assert result["arm"] == {"episodes": 1, "steps": 1, "stale_rows_ignored": 1}


def test_modern_exit_gate_cannot_fall_back_to_unstamped_ledger(tmp_path):
    journal, steps = tmp_path / "journal.jsonl", tmp_path / "per_step.jsonl"
    row = _terminal()
    row.pop("run_id")
    _write_rows(journal, [row])
    _write_rows(steps, [_step()])
    with pytest.raises(SystemExit, match="lacks attempt/run_id"):
        assert_accepted_full_hit(journal, steps, {"arm"}, require_run_id=True)


def test_census_keeps_both_runs_and_updates_legacy_path(tmp_path):
    per_step = tmp_path / "per_step.jsonl"
    for run, results in (("r1", 7), ("r2", 12)):
        write_census(SimpleNamespace(run_id=run, worker_census={"worker": {"results": results}}), per_step)
    first = json.loads(Path(str(per_step) + ".workers.r1.json").read_text())
    second = json.loads(Path(str(per_step) + ".workers.r2.json").read_text())
    latest = json.loads(Path(str(per_step) + ".workers.json").read_text())
    assert first["workers"]["worker"]["results"] == 7
    assert second["workers"]["worker"]["results"] == 12
    assert latest == second


@pytest.mark.parametrize("change", [None, "library", "template"])
def test_fit_binds_bytes_loaded_before_computation(tmp_path, monkeypatch, change):
    analysis = Path(__file__).resolve().parents[2] / "exp/weighted_sum/analysis"
    monkeypatch.syspath_prepend(str(analysis))
    fit = importlib.import_module("lcw_fit_weights")
    library, template, cells = [tmp_path / p for p in ("library.pkl", "template.yaml", "cells.json")]
    entries = [SimpleNamespace(
        trajectory_id=f"traj-{t}", step_idx=step,
        query_keys={f: np.array([step, t, 1.0], dtype=np.float32) for f in fit.FIELDS},
        payload=SimpleNamespace(task_key="task", action_chunk=np.zeros((5, 7), dtype=np.float32)))
        for t in range(2) for step in range(3)]
    library.write_bytes(pickle.dumps({"entries": entries}))
    ss = {"field_similarity": {f: {"type": "cosine"} for f in fit.FIELDS},
          "score_normalization": {"type": "per_field", "fields": {
              f: {"method": "zscore", "params": {"mu": 0.0, "sigma": 1.0, "squash": "tanh"}}
              for f in fit.FIELDS}}}
    template.write_text(yaml.safe_dump({"checkpoints": {"cp1": {"search_strategy": ss}}}))
    cells.write_text(json.dumps({"cells": [{"w": [1 / 3] * 3, "sr": 0.5, "n": 500}]}))
    changed = False

    def scores(keys, *args):
        nonlocal changed
        if change and not changed:
            path = library if change == "library" else template
            path.write_bytes(path.read_bytes() + (b"changed" if change == "library" else b"\n# changed\n"))
            changed = True
        return np.zeros((len(keys), len(keys)))

    raw, weights = [1.0, 2.0, 3.0], [1 / 6, 2 / 6, 3 / 6]
    monkeypatch.setattr(fit, "normalized_scores", scores)
    monkeypatch.setattr(fit, "fisher_and_lda", lambda *a: {
        "fisher_raw": raw, "fisher": weights, "lda_raw": raw, "lda": weights, "queries": 6})
    monkeypatch.setattr(fit, "pairwise_logistic", lambda *a: {"raw": raw, "w": weights})
    monkeypatch.setattr(fit.torch.cuda, "is_available", lambda: False)
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["fit", "--library", str(library), "--template", str(template),
                                      "--cells", str(cells), "--output", str(output), "--pairs", "12"])
    if change:
        with pytest.raises(SystemExit, match="changed during fitting"):
            fit.main()
        assert not (output / "lcw_fit_weights.json").exists()
    else:
        fit.main()
        result = json.loads((output / "lcw_fit_weights.json").read_text())
        assert result["library_sha256"] == fit.sha256_file(library)
        assert result["template"]["sha256"] == fit.sha256_file(template)
