"""Synthetic provenance and completion timestamps; no real A data."""
import ast
import json
from pathlib import Path

import pytest

from . import analyze as b, analyze_test_a as a
from .test_test_a_stages import synthetic, dump


def set_time(s, name, timestamp):
    from datetime import datetime, timezone
    path = s['out'] / name
    value = json.loads(path.read_bytes())
    value['analyzed_at_utc'] = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    dump(path, value)


def update_equivalence(s, change):
    path = s['out'] / 'partial_equivalence.json'
    eq = json.loads(path.read_bytes())
    change(eq)
    dump(path, eq)
    prov_path = s['out'] / 'provenance.json'
    prov = json.loads(prov_path.read_bytes())
    prov['selected_output_sha256'][path.name] = b.sha(path.read_bytes())
    dump(prov_path, prov)


@pytest.mark.parametrize('model', ('pi05', 'groot'))
def test_early_partial_later_full_with_equivalence_passes(synthetic, model):
    s = synthetic
    set_time(s, 'provenance.json', 5000.)  # Later than the first A terminal at 4000.
    a.main(['apply', '--stage', str(s['stages'][model])])
    result = json.loads((s['out'] / f'test_a_{model}.json').read_bytes())
    evidence = result['provenance']['freeze_before_test']
    assert set(evidence) == {model}
    assert evidence[model]['source'] == str(s['out'] / f'provenance_{model}.json')
    assert evidence[model]['timestamp'] == 1500. < evidence[model]['earliest_new_arm_completion_ts'] == 4000.
    assert evidence[model]['equivalence_required'] and evidence[model]['strict_freeze_before_test']


@pytest.mark.parametrize('timestamp', (4000., 4500.))
@pytest.mark.parametrize('model', ('pi05', 'groot'))
def test_own_partial_freeze_equal_or_later_than_first_A_fails(synthetic, model, timestamp):
    s = synthetic
    set_time(s, f'provenance_{model}.json', timestamp)
    set_time(s, 'provenance.json', 5000.)
    with pytest.raises(ValueError, match='B analysis not earlier than A completion records: ' + model):
        a.main(['apply', '--stage', str(s['stages'][model])])
    assert not (s['out'] / f'test_a_{model}.json').exists()


@pytest.mark.parametrize('missing', ('file', 'model', 'primary_check', 'sha', 'failed'))
def test_later_full_requires_model_equivalence(synthetic, missing):
    s = synthetic
    set_time(s, 'provenance.json', 5000.)
    if missing == 'file':
        (s['out'] / 'partial_equivalence.json').unlink()
        # Exercise the explicit chronology prerequisite as well as the existing
        # full-output hash guard (which already rejects an indexed missing file).
        path = s['out'] / 'provenance.json'
        prov = json.loads(path.read_bytes())
        del prov['selected_output_sha256']['partial_equivalence.json']
        dump(path, prov)
        message = 'Missing partial/full equivalence'
    else:
        def change(eq):
            if missing == 'model':
                eq['checked_models'] = ['groot']
            elif missing == 'primary_check':
                eq['checks'] = [c for c in eq['checks'] if c['file'] != 'levels_pi05.json']
            elif missing == 'sha':
                next(c for c in eq['checks'] if c['file'] == 'levels_pi05.json')['sha256'] = b.sha(b'unrelated')
            else:
                eq['PASS'] = False
        update_equivalence(s, change)
        message = 'equivalence'
    with pytest.raises(ValueError, match=message):
        a.main(['apply', '--stage', str(s['stages']['pi05'])])
    assert not s['reads'] and not (s['out'] / 'test_a_pi05.json').exists()


def test_equivalence_must_be_bound_to_full_provenance(synthetic):
    s = synthetic
    set_time(s, 'provenance.json', 5000.)
    path = s['out'] / 'provenance.json'
    prov = json.loads(path.read_bytes())
    del prov['selected_output_sha256']['partial_equivalence.json']
    dump(path, prov)
    with pytest.raises(ValueError, match='equivalence is not bound'):
        a.main(['apply', '--stage', str(s['stages']['pi05'])])
    assert not s['reads']


@pytest.mark.parametrize('missing', ('provenance', 'wrong_model', 'changed_table'))
def test_actual_model_freeze_required_and_bound_to_tables(synthetic, missing):
    s = synthetic
    path = s['out'] / 'provenance_pi05.json'
    if missing == 'provenance':
        path.unlink()
        message = 'Missing model freeze provenance'
    elif missing == 'wrong_model':
        prov = json.loads(path.read_bytes())
        prov['analysis_scope']['models'] = ['groot']
        dump(path, prov)
        message = 'Wrong model freeze provenance'
    else:
        dump(s['out'] / 'levels_pi05.json', {})
        message = 'Frozen partial B output changed'
    with pytest.raises(ValueError, match=message):
        a.main(['apply', '--stage', str(s['stages']['pi05'])])
    assert not s['reads']


def test_every_arm_retains_strict_order_check(synthetic):
    s = synthetic
    # A minimum alone could hide a NaN after a valid timestamp; the original
    # all-arm comparison rejects this invalid completion record.
    s['arms']['pi05_l10_500']['task_descent'].audit['earliest_accepted_terminal_ts'] = float('nan')
    with pytest.raises(ValueError, match='B analysis not earlier than A completion records: pi05'):
        a.main(['apply', '--stage', str(s['stages']['pi05'])])


def test_combination_checks_each_models_own_freeze(synthetic):
    s = synthetic
    set_time(s, 'provenance.json', 8000.)
    set_time(s, 'provenance_groot.json', 6000.)
    for cell, arms in s['arms'].items():
        if cell.startswith('groot_'):
            for label in ('uniform', 'task_descent'):
                arms[label].audit['earliest_accepted_terminal_ts'] = 7000.
        arms['pure_c10'].audit['earliest_accepted_terminal_ts'] = 10.
    a.main(['combine', '--stages', str(s['stages']['groot']), str(s['stages']['pi05'])])
    result = json.loads((s['out'] / 'test_a.json').read_bytes())
    evidence = result['provenance']['freeze_before_test']
    assert evidence['pi05']['timestamp'] == 1500. and evidence['pi05']['earliest_new_arm_completion_ts'] == 4000.
    assert evidence['groot']['timestamp'] == 6000. and evidence['groot']['earliest_new_arm_completion_ts'] == 7000.
    # A GR00T freeze earlier than its own A but later than pi05 A remains valid.
    assert evidence['groot']['timestamp'] > evidence['pi05']['earliest_new_arm_completion_ts']
    set_time(s, 'provenance_groot.json', 7500.)
    with pytest.raises(ValueError, match='B analysis not earlier than A completion records: groot'):
        a.main(['combine', '--stages', str(s['stages']['pi05']), str(s['stages']['groot'])])


@pytest.mark.parametrize('timestamp', (4000., 5000.))
def test_legacy_full_stage_keeps_full_provenance_order(synthetic, timestamp):
    s = synthetic
    set_time(s, 'provenance.json', timestamp)
    with pytest.raises(ValueError, match='B analysis not earlier than A completion records: full'):
        a.main(['apply', '--stage', str(s['stages']['full'])])
    assert not (s['out'] / 'test_a.json').exists()


def test_legacy_full_stage_does_not_require_partial_provenance(synthetic):
    s = synthetic
    (s['out'] / 'provenance_pi05.json').unlink()
    (s['out'] / 'provenance_groot.json').unlink()
    a.main(['apply', '--stage', str(s['stages']['full'])])
    result = json.loads((s['out'] / 'test_a.json').read_bytes())
    assert set(result['provenance']['freeze_before_test']) == {'full'}


def test_freeze_order_fix_leaves_all_other_existing_functions_identical():
    root = Path(a.__file__).parent
    functions = lambda data: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(data).body
                              if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    before = functions((root / 'analyze_test_a.before_freeze_order_fix.py.txt').read_bytes())
    after = functions(Path(a.__file__).read_bytes())
    assert all(before[name] == after[name] for name in before if name != 'apply')
