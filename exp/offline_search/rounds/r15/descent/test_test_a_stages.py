"""Synthetic-only stage scopes, output isolation and frozen pooling equivalence."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from . import analyze as bmod, analyze_test_a as a


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bmod.output_text(value))


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    out = tmp_path / 'outputs'
    out.mkdir()
    cells = tuple(a.CELLS)
    coefficients = {'pi05': [.2, .8], 'groot': [.7, .3]}
    p = dict(frozen_at_utc='synthetic-pre-outcome', cost=dict(coefficients=coefficients))
    levels = {c: {str(t): ('ir32' if t % 2 else 'off') for t in range(10)} for c in cells}
    uniforms = {c: 'ir40' for c in cells}
    rng = np.random.default_rng(481)
    arms, b, shared = {}, {'cells': {}}, {}
    for cell in cells:
        model, suite, size = cell.split('_')
        cv, cm = coefficients[model]
        full = dict(s1=cv, s2=cm / 2, s3=cm / 2, k=10 if model == 'pi05' else 8)
        arms[cell] = {}
        for j, arm in enumerate(a.ARMS):
            key = (model, suite)
            if arm == 'pure_c10' and key in shared:
                arms[cell][arm] = shared[key]
                continue
            # Different cell decision counts expose incorrect averaging of IR;
            # differing paired outcomes expose broken suite-shared resampling.
            n = rng.integers(3 + 9 * (model == 'groot'), 12 + 9 * (model == 'groot'), (10, 50))
            v = rng.binomial(n, [.8, .5, .7][j])
            m = rng.binomial(n, [.7, .6, .8][j])
            y = rng.binomial(1, .72 - .03 * j + .04 * (size == '500'), (10, 50))
            ep = a.Episodes(y, n, v, m, cv * v + cm * m,
                            dict(full_stage_costs=full, earliest_accepted_terminal_ts=4000.))
            arms[cell][arm] = ep
            if arm == 'pure_c10':
                shared[key] = ep
        # A different development SR/cost mixture exercises selection optimism.
        stats = {}
        for arm, ep in arms[cell].items():
            y, n, v, m, cost = (getattr(ep, k)[:, :30] for k in ('y', 'n', 'v', 'm', 'cost'))
            stats[arm] = a.Episodes(y, n, v, m, cost, {}).stats()
        stats['pure_c10']['full_stage_costs'] = full
        b['cells'][cell] = dict(arm_stats={'c10': stats['pure_c10']}, matched_ir={'synthetic': cell},
            tolerances={'5': dict(levels=levels[cell], uniform=uniforms[cell], library={'synthetic': size},
                savings={'uniform': stats['uniform'], 'task': stats['task_descent']})})
    files = {'analysis.json': b, 'levels.json': levels, 'uniform.json': uniforms}
    for name, value in files.items():
        dump(out / name, value)
    checks = []
    for model in ('pi05', 'groot'):
        selected = [c for c in cells if c.startswith(model + '_')]
        hashes = {}
        for kind, table in (('levels', levels), ('uniform', uniforms)):
            name = f'{kind}_{model}.json'
            dump(out / name, {c: table[c] for c in selected})
            hashes[name] = bmod.sha((out / name).read_bytes())
            checks.append(dict(model=model, file=name, sha256=hashes[name], byte_identical=True))
        dump(out / f'provenance_{model}.json', dict(analyzed_at_utc='1970-01-01T00:25:00+00:00',
            outcome_pool='B', analysis_scope=dict(partial=True, models=[model], cells=selected),
            selected_output_sha256=hashes))
    dump(out / 'partial_equivalence.json', dict(PASS=True, checked_models=['pi05', 'groot'], checks=checks))
    dump(out / 'provenance.json', dict(analyzed_at_utc='1970-01-01T00:33:20+00:00',
         selected_output_sha256={name: bmod.sha((out / name).read_bytes())
                                for name in (*files, 'partial_equivalence.json')}))
    dump(out / 'cap_tables.json', {'synthetic': True})
    (out / 'analyze.py').write_text('# synthetic B analyzer\n')
    trace = tmp_path / 'trace'
    for i in range(1, 5):
        dump(trace / f'r11_knob_{i}/eval500.json', a.EXPECTED)
    stages = {}
    for model in ('pi05', 'groot', 'full'):
        stage = out / ('stage_' + model)
        selected = [c for c in cells if model == 'full' or c.startswith(model + '_')]
        dump(stage / 'input.json', dict(levels={c: levels[c] for c in selected},
                                       uniform={c: uniforms[c] for c in selected}))
        stages[model] = stage
    reads = []

    def load(cell, selected_levels, uniform, protocol, caps, reader, template):
        assert selected_levels == levels[cell] and uniform == uniforms[cell]
        assert protocol == p and caps == {'synthetic': True}
        assert json.loads(template) == [list(x) for x in a.EXPECTED]
        reads.append(cell)
        return arms[cell]

    monkeypatch.setattr(a, 'HERE', out)
    monkeypatch.setattr(a, 'TRACE', trace)
    monkeypatch.setattr(a, 'EMITTER', stages['full'])
    monkeypatch.setattr(a, 'verify_protocol', lambda reader: p)
    monkeypatch.setattr(a, 'load_a', load)
    monkeypatch.setattr(a, 'DRAWS', 128)  # Fixed production seed/draw count is checked by AST tests.
    return dict(out=out, cells=cells, stages=stages, levels=levels, uniforms=uniforms,
                arms=arms, b=b, p=p, reads=reads)


@pytest.mark.parametrize('model', ('pi05', 'groot'))
def test_complete_subset_accepted_and_outputs_isolated(synthetic, model):
    s = synthetic
    (s['out'] / 'TEST_A.md').write_text('existing full report\n')
    (s['out'] / 'test_a.json').write_text('existing full result\n')
    a.main(['apply', '--stage', str(s['stages'][model])])
    result = json.loads((s['out'] / f'test_a_{model}.json').read_text())
    expected = [c for c in s['cells'] if c.startswith(model + '_')]
    assert s['reads'] == expected and list(result['cells']) == expected
    assert result['analysis_scope']['partial'] and not result['analysis_scope']['combined']
    assert result['levels'] == {c: s['levels'][c] for c in expected}
    for size, group in result['groups_by_library_size'].items():
        assert group['pooled']['arms']['pure_c10']['n_episodes'] == 1000
        assert group['partial'] and 'partial single-model' in group['methods']['pooled_sr']
    text = (s['out'] / f'TEST_A_{model}.md').read_text()
    assert 'Partial measurement' in text and 'Forty task comparisons' in text
    handback = (s['out'] / f'ANALYSIS_TEST_A_HANDBACK_{model}.md').read_text()
    assert f'TEST_A_{model}.md' in handback and f'test_a_{model}.json' in handback
    assert (s['out'] / f'REPORT_{model}.md').exists()
    assert (s['out'] / 'TEST_A.md').read_text() == 'existing full report\n'
    assert (s['out'] / 'test_a.json').read_text() == 'existing full result\n'
    full = a.calculate(s['arms'], s['b'], s['p'], s['levels'], s['uniforms'])
    assert result['cells'] == {c: full['cells'][c] for c in expected}


@pytest.mark.parametrize('change, message', (
    ('level', 'Emitter input differs from B'),
    ('uniform', 'Emitter input differs from B'),
    ('missing', 'complete model cell set'),
    ('mixed', 'complete model cell set'),
    ('different_keys', 'Stage level/uniform cells differ'),
))
def test_mismatched_subset_rejected_before_outcome_reads(synthetic, change, message):
    s = synthetic
    path = s['stages']['pi05'] / 'input.json'
    data = json.loads(path.read_text())
    cell = next(iter(data['levels']))
    if change == 'level':
        data['levels'][cell]['0'] = 'ir25'
    elif change == 'uniform':
        data['uniform'][cell] = 'off'
    elif change == 'missing':
        del data['levels'][cell], data['uniform'][cell]
    elif change == 'different_keys':
        del data['uniform'][cell]
    else:
        other = cell.replace('pi05', 'groot')
        data['levels'][other] = data['levels'].pop(cell)
        data['uniform'][other] = data['uniform'].pop(cell)
    dump(path, data)
    with pytest.raises(ValueError, match=message):
        a.main(['apply', '--stage', str(s['stages']['pi05'])])
    assert not s['reads'] and not (s['out'] / 'test_a_pi05.json').exists()


def measurement(result):
    return {k: v for k, v in result.items() if k not in ('provenance', 'analysis_scope')}


def test_combined_equals_full_stage_and_frozen_calculation(synthetic, monkeypatch):
    s = synthetic
    monkeypatch.setattr(a, 'DRAWS', 5000)
    # No partial summaries exist: combination must audit and pool raw episodes.
    a.main(['combine', '--stages', str(s['stages']['groot']), str(s['stages']['pi05'])])
    combined = json.loads((s['out'] / 'test_a.json').read_text())
    combined_markdown = (s['out'] / 'TEST_A.md').read_bytes()
    assert s['reads'] == list(s['cells'])  # Stage order cannot change summation/resampling order.
    assert combined['analysis_scope']['combined'] and not combined['analysis_scope']['partial']
    s['reads'].clear()
    a.main(['apply', '--stage', str(s['stages']['full'])])
    full = json.loads((s['out'] / 'test_a.json').read_text())
    assert measurement(combined) == measurement(full)
    assert combined_markdown == (s['out'] / 'TEST_A.md').read_bytes()
    assert s['reads'] == list(s['cells'])
    frozen = a.calculate(s['arms'], s['b'], s['p'], s['levels'], s['uniforms'])
    # Compare to the untouched frozen calculation, not only two new entrypoints.
    assert {k: v for k, v in measurement(full).items() if k != 'episode_data'} == frozen
    for size, group in combined['groups_by_library_size'].items():
        stats = group['pooled']['arms']['pure_c10']
        assert stats['n_episodes'] == 2000
        rows = [c['arms']['pure_c10'] for c in group['cells'].values()]
        expected_cost = sum(s['p']['cost']['coefficients'][c.split('_')[0]][0] * r['arms']['pure_c10']['vision'] +
                            s['p']['cost']['coefficients'][c.split('_')[0]][1] * r['arms']['pure_c10']['misses']
                            for c, r in group['cells'].items())
        assert stats['ir'] == expected_cost / sum(r['decisions'] for r in rows)
        assert '2000 episode outcomes' in group['methods']['pooled_sr']
    assert a.CELLS == s['cells']


@pytest.mark.parametrize('models', (('pi05', 'pi05'), ('full', 'groot')))
def test_combination_rejects_duplicate_or_full_stage(synthetic, models):
    s = synthetic
    with pytest.raises(ValueError, match='disjoint complete pi05 and groot'):
        a.main(['combine', '--stages', *(str(s['stages'][m]) for m in models)])
    assert not s['reads']


@pytest.mark.parametrize('failure', ('b_hash', 'chronology', 'cost_spec', 'templates'))
def test_existing_checks_preserved(synthetic, failure):
    s = synthetic
    if failure == 'b_hash':
        (s['out'] / 'levels.json').write_text('{}\n')
        message = 'Frozen B output changed'
    elif failure == 'chronology':
        for cell in s['cells']:
            for arm in ('uniform', 'task_descent'):
                s['arms'][cell][arm].audit['earliest_accepted_terminal_ts'] = 1000.
        message = 'B analysis not earlier'
    elif failure == 'cost_spec':
        cell = s['cells'][0]
        s['arms'][cell]['uniform'].audit['full_stage_costs'] = {'wrong': 1}
        message = 'A full-policy cost specification differs'
    else:
        dump(a.TRACE / 'r11_knob_4/eval500.json', ['different'])
        message = 'R11 A templates differ'
    with pytest.raises(ValueError, match=message):
        a.main(['apply', '--stage', str(s['stages']['pi05'])])
    assert not (s['out'] / 'test_a_pi05.json').exists()


def test_frozen_loader_and_all_statistics_ast_identical():
    original = (Path(a.__file__).parent / 'test_a_stage_original_analyze_test_a.py.txt').read_bytes()
    current = Path(a.__file__).read_bytes()
    functions = lambda data: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(data).body
                              if isinstance(n, ast.FunctionDef)}
    before, after = functions(original), functions(current)
    assert all(before[name] == after[name] for name in a.UNCHANGED_FUNCTIONS)
    original_namespace = {}
    exec(compile(ast.Module(body=[n for n in ast.parse(original).body if isinstance(n, ast.Assign) and
                                 any(isinstance(t, ast.Tuple) for t in n.targets)], type_ignores=[]),
                 '<frozen constants>', 'exec'), {}, original_namespace)
    assert (a.SEED, a.DRAWS) == (original_namespace['SEED'], original_namespace['DRAWS'])


def test_registered_protocol_verifies_without_outcome_reads():
    reader = a.Reader()
    p = a.verify_protocol(reader)
    reader.stable()
    source = str(Path(a.__file__).resolve())
    assert reader.inputs[source]['sha256'] == bmod.sha(Path(source).read_bytes())
    assert p['input_sha256'][str(Path(source).relative_to(a.REPO))] != reader.inputs[source]['sha256']
    assert not reader.shards
    assert not any('/runs/' in path or '/test_a_' in path and '/stage_' in path for path in reader.inputs)


@pytest.mark.parametrize('change, message', (
    ('unregistered', 'Unregistered test-A source change'),
    ('numerical', 'Frozen test-A measurement/loader function changed'),
    ('other_input', 'Frozen input changed'),
))
def test_adapter_rejects_unregistered_or_numerical_or_other_source_changes(change, message):
    source = Path(a.__file__).resolve()
    actual = source.read_bytes()
    changed = actual.replace(b'family=40)', b'family=39)') if change == 'numerical' else actual + b'\n# unregistered\n'
    assert changed != actual

    class AlteredReader(a.Reader):
        def read(self, path):
            data = super().read(path)
            path = Path(path).resolve()
            if change in ('unregistered', 'numerical') and path == source:
                return changed
            if change == 'numerical' and path == a.HERE / 'test_a_stage_adapter.json':
                record = json.loads(data)
                record['adapted_sha256'] = bmod.sha(changed)
                return json.dumps(record).encode()
            if change == 'other_input' and path == a.HERE / 'prepare.py':
                return data + b'\n# changed input\n'
            return data

    with pytest.raises(ValueError, match=message):
        a.verify_protocol(AlteredReader())
