"""Synthetic outcomes only; pytest scratch must be outside the repository."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from . import analyze, build
from .runtime import LoggedDistanceBase, PureCorrector
from exp.offline_search.rounds.r10.astra.method import DistanceCorrectedBase, multiplier


def test_switches_are_boolean_and_runtime_never_refits():
    with pytest.raises(ValueError, match='booleans'):
        PureCorrector(corrector_enabled=1)
    with pytest.raises(RuntimeError, match='refitting is forbidden'):
        PureCorrector().fit(None, None)


def test_cross_size_assembly_is_refused():
    with pytest.raises(ValueError, match='matching frozen'):
        PureCorrector.assemble(SimpleNamespace(variant='A', size=50),
                              SimpleNamespace(variant='GC_dist', size=100))


@pytest.mark.parametrize('ratio,step,strength', [(0., 0, 0.), (.75, 1, .5), (1., 1, .4),
                                               (1.5, 1, .2), (2., 1, 0.), (np.inf, 1, 0.)])
def test_frozen_strength(ratio, step, strength):
    assert .5 * multiplier(ratio, step, dict(peak=.5, plateau=.75, cutoff=2.)) == pytest.approx(strength)


def test_telemetry_preserves_returned_correction(monkeypatch):
    corr = np.full((10, 6), .25, np.float32)
    monkeypatch.setattr(DistanceCorrectedBase, '_correction', lambda self, q, action: corr)
    base = LoggedDistanceBase.__new__(LoggedDistanceBase)
    base.blend, base.distance_scale = .5, 2.
    base.distance_rule = dict(peak=.5, plateau=.75, cutoff=2.)
    base._anchor = dict(rows=np.array([4]))
    base._dist = lambda q: (SimpleNamespace(rows=np.array([4])), 1, 2, None, None, None,
                           None, np.array([2.]), None, None, None)
    out = base._correction(SimpleNamespace(step=1), None)
    assert out is corr
    assert base.pc_log['os_pc_strength'] == pytest.approx(.4)
    assert base.pc_log['os_pc_applied'] == base.pc_log['os_pc_count'] == 1.


def test_exact_pairing_direction_and_concordance():
    assert analyze.paired([True] * 10, [False] * 10)['p_two_sided_exact'] == 2 / 2**10
    p = analyze.paired([False] * 10, [True] * 10)
    assert (p['gains'], p['losses'], p['delta_sr']) == (0, 10, -1.)
    assert analyze.paired([True, False], [True, False])['p_two_sided_exact'] == 1.


def fixture_arm(root, arm, y, size, *, model='pi05', ir=.076):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'eval500.json'
    manifest.write_text(json.dumps(build.EXPECTED) + '\n')
    row = dict(arm=arm, model=model, manifest=str(manifest), replan_steps=5,
               recording=False, rollout_replicate_seed=7)
    rows = json.loads((root / 'arms.json').read_text()) if (root / 'arms.json').exists() else []
    (root / 'arms.json').write_text(json.dumps(rows + [row]))
    run = root / 'runs' / arm
    (run / 'client').mkdir(parents=True)
    (run / 'manifest.json').write_bytes(manifest.read_bytes())
    journal = [dict(task_uid=f'{arm}:libero_10:{t}:{i}', accepted=True, status='done',
                    success=bool(y[j]), init_pool='A') for j, (t, i) in enumerate(build.EXPECTED)]
    (run / 'client/journal.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in journal))
    summary = run / 'summary.json'
    summary.write_text(json.dumps(dict(arm=arm, complete=500, success=sum(y), sr=sum(y) / 500,
                                       cost_ledger=dict(ir_per_five_controls=ir, controls=size * 500))))
    return dict(root=str(root), arm=arm, files=dict(manifest=dict(path=str(manifest), sha256=build.sha(manifest)),
                   summary=dict(path=str(summary), sha256=build.sha(summary))))


@pytest.fixture
def synthetic_design(tmp_path):
    records = []
    for model, suite in build.CELLS:
        for size in build.SIZES:
            y = [i < 200 for i in range(500)]
            yc = [5 <= i < 220 for i in range(500)]
            root = tmp_path / f'{model}_{suite}'
            refs = {v: fixture_arm(root, f'{model}_{suite}_{size}_{v}',
                    y if v == 'A' else [i < 210 for i in range(500)], size, model=model)
                    for v in ('A', 'G', 'GC_dist')}
            new = fixture_arm(root, f'{model}_{suite}_{size}_AC', yc, size, model=model, ir=.06)
            records.append(dict(model=model, suite=suite, size=size, root=new['root'], arm=new['arm'], references=refs))
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(PASS=True, records=records)))
    return audit, records


def test_complete_synthetic_analysis_and_size_pooled_pairing(synthetic_design):
    audit, _ = synthetic_design
    groups = analyze.analyze(audit)
    assert len(groups) == 4
    for group in groups:
        assert len(group['cells']) == 6
        pooled = group['pooled_pairing']
        assert (pooled['n'], pooled['gains'], pooled['losses']) == (3000, 120, 30)
        for c in group['cells']:
            assert (c['pairing']['gains'], c['pairing']['losses']) == (20, 5)
            assert c['arms']['A+C']['sr'] == .43
            assert c['arms']['A+C']['ir'] == .06
        assert group['pooled_arms']['A+C']['ir'] == pytest.approx(.06)
    rendered = analyze.render(groups)
    assert rendered.count('pooled (3000)') == 4
    assert '0.400 @ 0.076' in rendered and '+20/-5' in rendered


def test_analysis_refuses_missing_pair(synthetic_design):
    audit, records = synthetic_design
    r = records[0]
    p = Path(r['root']) / 'runs' / r['arm'] / 'client/journal.jsonl'
    p.write_text('\n'.join(p.read_text().splitlines()[1:]) + '\n')
    with pytest.raises(ValueError, match='missing, extra'):
        analyze.analyze(audit)


def test_analysis_refuses_changed_template(synthetic_design):
    audit, records = synthetic_design
    r = records[0]
    p = Path(r['root']) / 'runs' / r['arm'] / 'manifest.json'
    p.write_text(json.dumps(build.EXPECTED[:-1]))
    with pytest.raises(ValueError, match='identities/template'):
        analyze.analyze(audit)


def test_audit_refuses_changed_sha_before_loading_any_fit(monkeypatch, tmp_path):
    path = tmp_path / 'source'
    path.write_text('changed')
    fake = dict(arm='arm', references={'A': {'files': {'fit': dict(path=str(path), sha256='0' * 64)}}})
    records = [dict(fake, arm=f'arm{i}') for i in range(24)]
    (tmp_path / 'audit.json').write_text(json.dumps(dict(records=records)))
    monkeypatch.setattr(build, 'HERE', tmp_path)
    with pytest.raises(ValueError, match='changed R10 reference'):
        build.audit()


def test_plans_refuse_wrong_fleet_before_planning(monkeypatch):
    monkeypatch.setenv('SERVER_HOST', 'h100')
    monkeypatch.setenv('WORKER_HOST', 'timan108')
    with pytest.raises(ValueError, match='timan107'):
        build.plans()
