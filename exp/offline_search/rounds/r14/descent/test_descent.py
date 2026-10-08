"""Actual plugin decision equivalence, off/C10 endpoints and emitter installation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace as NS

import numpy as np
import pytest
from exp.offline_search.closed_loop import plugin
from exp.offline_search.rounds.r14.impl.mechanics import Harness, equal_result
from exp.offline_search.rounds.r14.knob.controller import propose
from .prepare import (CELLS, LEVELS, DEV_ORDER, TRACE, frozen, level_cap, descent_blob,
                      emit_test_a, install_test_a, check_seal)
from .verify import verify_records


def load(cell, level):
    return pickle.loads(Path(frozen(cell, level)['path']).read_bytes())['method']


def lifecycle(rows):
    return [(r['task_id'], r['step'], r['hit'], r['vision'], r['src'], r['served_head']) for r in rows]


@pytest.mark.parametrize('cell', CELLS)
def test_uniform_all_levels_actual_plugin(cell, tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, '_git_head', lambda: None)
    h = Harness(cell, tmp_path / cell)
    try:
        for level in LEVELS:
            uniform = load(cell, level)
            method = descent_blob(cell, {str(t): level for t in range(10)})['method']
            assert method.knob_settings['K'] == [level_cap(cell, level)] * 10
            reference = h.run(uniform, tasks=range(10), steps=12, force_guards=True)
            actual = h.run(method, tasks=range(10), steps=12, force_guards=True, retries=level != 'off')
            assert reference[0] == actual[0], (cell, level, 'action bytes')
            assert lifecycle(reference[2]) == lifecycle(actual[2]), (cell, level, 'decisions')
            if level == 'off':
                assert all(not r['r14'].get('knob', False) for r in actual[2])
            elif level == 'c10':
                assert all(not r['vision'] or not r['hit'] for r in actual[2])
            else:
                ref = [r['r14'] for r in reference[2] if r['vision']]
                out = [r['r14'] for r in actual[2] if r['vision']]
                assert [(r['cap'], r['coin'], r['knob'], r['guard']) for r in ref] == [(r['cap'], r['coin'], r['knob'], r['guard']) for r in out]
    finally:
        h.rt.flush_all()


@pytest.mark.parametrize('cell', CELLS)
def test_mixed_levels_actual_plugin(cell, tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, '_git_head', lambda: None)
    h = Harness(cell, tmp_path / cell)
    levels = {str(t): LEVELS[t % len(LEVELS)] for t in range(10)}
    method = descent_blob(cell, levels)['method']
    try:
        actual = h.run(method, tasks=range(10), steps=12, reverse=True, force_guards=True)
        for level in LEVELS:
            tasks = [t for t in range(10) if levels[str(t)] == level]
            ref = h.run(load(cell, level), tasks=tasks, steps=12, force_guards=True)
            actions = {k: v for k, v in actual[0].items() if k[0] in tasks}
            assert actions == ref[0]
            rows = [r for r in actual[2] if r['task_id'] in tasks]
            assert sorted(lifecycle(rows)) == sorted(lifecycle(ref[2]))
    finally:
        h.rt.flush_all()


def test_controller_long_history_endpoints():
    # The endpoint is independent of episode length; off is an explicit branch.
    for task in range(10):
        for count in (0, 1, 7, 10000):
            q = NS(step=count, task_id=task, episode=NS(task_id=task, init=49),
                   hist_has_vision=np.ones(count, bool), hist_hit=np.ones(count, int))
            assert propose(q, False, [0.] * 10)['knob']
            assert not propose(q, True, [0.] * 10)['knob']


def test_emitter_install_and_contracts(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin, '_git_head', lambda: None)
    levels = {cell: {str(t): LEVELS[t % len(LEVELS)] for t in range(10)} for cell in CELLS}
    uniform = {cell: 'ir32' for cell in CELLS}
    stage = tmp_path / 'stage'
    records = emit_test_a(levels, uniform, stage)
    assert len(records) == 4
    verify_records(records)
    original = (TRACE / 'r11_knob_1/eval500.json').read_bytes()
    for rec in records:
        assert (Path(rec['root']) / 'eval500.json').read_bytes() == original
        arm = next(r for r in json.loads((Path(rec['root']) / 'arms.json').read_text())
                   if r['r14_descent_level'] == 'task_descent')
        args = ['--os-method', arm['method'], '--os-kwargs', json.dumps(arm['kwargs']),
                '--os-cell', arm['cell'], '--os-log-dir', str(tmp_path / 'load' / rec['cell']),
                '--os-tag', 'r14_descent_load', *arm['plugin_args']]
        opts, rest = plugin.parse_cli(args)
        assert not rest
        rt = plugin.PluginRuntime(opts, arm['model'])
        try:
            assert rt.method.task_levels == levels[rec['cell']]
            assert rt.method.knob_settings['K'] == arm['kwargs']['task_caps']
        finally:
            rt.flush_all()
    targets = install_test_a(stage, tmp_path / 'installed')
    for target in targets:
        check_seal(target)
        assert (Path(target) / 'eval500.json').read_bytes() == original
        assert all(row['manifest'] == str(Path(target) / 'eval500.json') for row in json.loads((Path(target) / 'arms.json').read_text()))
    verify_records([check_seal(p) for p in targets])
    with pytest.raises(FileExistsError):
        install_test_a(stage, tmp_path / 'installed')
    (Path(records[0]['root']) / 'arms.json').write_text('[]')
    with pytest.raises(ValueError, match='changed root definition'):
        install_test_a(stage, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


@pytest.mark.parametrize('bad', [{'0': 'off'}, {str(t): 'bad' for t in range(10)}])
def test_invalid_table_refused(bad):
    with pytest.raises(ValueError):
        descent_blob(CELLS[0], bad)
