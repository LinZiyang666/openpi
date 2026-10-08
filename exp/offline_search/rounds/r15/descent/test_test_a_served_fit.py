"""Synthetic A fits and logs only; both serving branches and strict rejection."""
import ast
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from exp.offline_search.closed_loop.ops.h100.assets import flag, remap
from exp.offline_search.rounds.r15.descent.synthetic_pickle_memo import SyntheticPickleMemo
from . import analyze as b, analyze_test_a as a
from .test_analysis import dump, fake_run, jl, seal
from . import verify_test_a_fit_hashes as audit
from .verify_test_a_fit_hashes import decision_paths_hashes


def fake_a(tmp_path, monkeypatch, model='pi05', remote=False, location='external', batch=False):
    cell = model + '_l10_100'
    monkeypatch.setattr(a, 'TRACE', tmp_path)
    levels = {str(t): 'ir32' for t in range(10)}
    uniform = 'ir32'
    store, base = tmp_path / 'store', Path('/custom/h100')
    sharedroot, target = tmp_path / 'shared_reference', tmp_path / ('r15_descent_test_' + cell)
    full = dict(s1=.152 if model == 'pi05' else .148,
                s2=.410, s3=.438 if model == 'pi05' else .442, k=10 if model == 'pi05' else 8)
    fitdict = {}
    for root in (sharedroot, target):
        root.mkdir(parents=True)
        manifest = root / 'eval500.json'
        dump(manifest, a.EXPECTED)
        rows = []
        sources = []
        for arm in ('pure_c10',) if root == sharedroot else ('uniform', 'task_descent'):
            name = f'r14dt_{model}_l10_50_pure_c10' if arm == 'pure_c10' else f'r15dt_{cell}_' + (
                'uniform_ir32' if arm == 'uniform' else arm)
            directory = root / 'artifacts' if location == 'internal' else store / 'fits' if location == 'store' else tmp_path / 'fits'
            fit = directory / (arm + '.pkl')
            fit.parent.mkdir(parents=True, exist_ok=True)
            blob = dict(store=str(store / 'library/synthetic'), run=str(root / 'runs' / name),
                        external='/home/weiland/synthetic/library.pkl',
                        values=[np.ones((2, 3), dtype=np.float32), np.zeros((3, 2), dtype=np.float32)])
            if batch:
                blob['pickle_memo'] = SyntheticPickleMemo()
            source = pickle.dumps(blob, protocol=4)
            fit.write_bytes(source)
            row = dict(arm=name, model=model, dev=False, init_pool='A', episodes=500,
                r14_descent_level='c10' if arm == 'pure_c10' else 'task_descent' if arm == 'task_descent' else uniform,
                replan_steps=5, full_model=True, recording=False, manifest=str(manifest),
                intended_host='h100' if remote else 'local', method='synthetic:FrozenController',
                plugin_args=['--os-fit-artifact', str(fit), '--os-root', str(store)],
                kwargs=dict(task_levels=levels, task_caps=[1.] * 10) if arm == 'task_descent' else dict(method='periodic'))
            rows.append(row)
            fake_run(root, row, a.EXPECTED, 'A')
            sources.append(source)
            served = (a.h100_relocated_batch_sha256(sources, root, store, base) if batch else
                      b.h100_relocated_sha256(source, root, store, base)) if remote else b.sha(source)
            served_path = remap(str(fit), root, store, base) if remote else str(fit)
            if remote:
                assert served != b.sha(source) and served_path != str(fit)
            run = root / 'runs' / name
            summary = json.loads((run / 'summary.json').read_bytes())
            summary['model'] = model
            summary['cost_ledger']['full_stage_costs'] = full
            dump(run / 'summary.json', summary)
            for shard in run.glob('server_*/decisions_*.jsonl'):
                decs = [json.loads(line) for line in shard.read_bytes().splitlines()]
                for d in decs:
                    d['r14']['fit_sha256'] = served
                    if d.get('miss_k') is not None:
                        d['miss_k'] = full['k']
                startup = dict(ev='startup', host='synthetic_h100' if remote else 'synthetic_local',
                    argv=['--os-fit-artifact', served_path], kwargs=row['kwargs'], method_spec=row['method'])
                jl(shard, [startup] + decs)
            fitdict[arm] = dict(path=str(fit), sha256=b.sha(source))
        dump(root / 'arms.json', rows)
        seal(root, cell, 'A', rows)
        if remote:
            # Only the last arm is retained, matching real sync's overwrite.
            dump(root / 'h100_sync/plan.json', dict(run=str(root), remote_run=str(base / 'runs' / root.name),
                                                 arms=rows if batch else [rows[-1]], files=[]))
    shared = dict(root=str(sharedroot), arm=f'r14dt_{model}_l10_50_pure_c10',
                  arms_sha256=b.sha((sharedroot / 'arms.json').read_bytes()), fit_sha256=fitdict['pure_c10']['sha256'])
    p = dict(test_references={model + '_l10': shared},
             cost=dict(coefficients={'pi05': [.152, .848], 'groot': [.148, .852]}))
    caps = dict(cells={cell: dict(periodic_cap_by_level={'ir32': 1.},
                                test_fit_sha256_by_level={'ir32': fitdict['uniform']['sha256']})})
    template = (target / 'eval500.json').read_bytes()
    dump(target / 'descent_table.json', dict(levels=levels, uniform=uniform, off_tasks=[], caps=[1.] * 10,
        template_sha256=b.sha(template), source_artifacts={'ir32': fitdict['uniform']['sha256']},
        shared_c10_reference=shared, fit=fitdict['task_descent']))
    return cell, levels, uniform, p, caps, template, target


@pytest.mark.parametrize('model', ('pi05', 'groot'))
def test_local_A_loader_branch_and_unchanged_measurements(tmp_path, monkeypatch, model):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch, model)
    loaded = a.load_a(cell, levels, uniform, p, caps, a.Reader(), template)
    original = a.HERE / 'analyze_test_a.before_served_fit_fix.py.txt'
    node = next(n for n in ast.parse(original.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'load_a')
    namespace = dict(vars(a))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen A loader>', 'exec'), namespace)
    before = namespace['load_a'](cell, levels, uniform, p, caps, a.Reader(), template)
    for arm, ep in loaded.items():
        for field in ('y', 'n', 'v', 'm', 'cost'):
            np.testing.assert_array_equal(getattr(ep, field), getattr(before[arm], field))
        assert {k: v for k, v in ep.audit.items() if k != 'fit_hash_verification'} == before[arm].audit
        audit = ep.audit['fit_hash_verification']
        assert set(audit['observed_path_branches'].values()) == {'frozen_source'}
        assert set(audit['observed_sha256_branches'].values()) == {'frozen_source'}
        assert audit['relocated_sha256'] is None


@pytest.mark.parametrize('location', ('external', 'internal', 'store'))
def test_h100_A_loader_all_arms_and_path_mapping(tmp_path, monkeypatch, location):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch, remote=True, location=location)
    reader = a.Reader()
    loaded = a.load_a(cell, levels, uniform, p, caps, reader, template)
    for arm, ep in loaded.items():
        assert ep.y.sum() == 499 and ep.n.sum() == ep.v.sum() == ep.m.sum() == 500
        assert ep.stats()['ir'] == pytest.approx(1.)
        audit = ep.audit['fit_hash_verification']
        assert set(audit['observed_path_branches'].values()) == {'h100_relocated'}
        assert set(audit['observed_sha256_branches'].values()) == {'h100_relocated'}
        assert audit['relocated_sha256'] != audit['source_sha256']
        expected_part = '/mirror/' if location == 'external' else '/runs/' if location == 'internal' else '/store/'
        assert expected_part in audit['relocated_path']
    reader.stable()


def test_h100_A_loader_reproduces_two_arm_sync_pickle_state(tmp_path, monkeypatch):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch, remote=True, batch=True)
    loaded = a.load_a(cell, levels, uniform, p, caps, a.Reader(), template)
    rows = json.loads((target / 'arms.json').read_bytes())
    uniform_audit = loaded['uniform'].audit['fit_hash_verification']
    task_audit = loaded['task_descent'].audit['fit_hash_verification']
    assert uniform_audit['relocation_arm_order'] == [rows[0]['arm']]
    assert task_audit['relocation_arm_order'] == [r['arm'] for r in rows]
    standalone = b.h100_relocated_sha256(Path(flag(rows[1]['plugin_args'], '--os-fit-artifact')).read_bytes(),
                                        target, tmp_path / 'store', Path('/custom/h100'))
    assert standalone != task_audit['relocated_sha256']
    assert set(task_audit['observed_sha256_branches'].values()) == {'h100_relocated'}


def test_h100_A_loader_rejects_unsealed_batch_arm(tmp_path, monkeypatch):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch, remote=True, batch=True)
    path = target / 'h100_sync/plan.json'
    plan = json.loads(path.read_bytes())
    plan['arms'][0]['arm'] = 'unrelated_unsealed_arm'
    dump(path, plan)
    with pytest.raises(ValueError, match='Unsealed H100 sync arm order'):
        a.load_a(cell, levels, uniform, p, caps, a.Reader(), template)


@pytest.mark.parametrize('change', ('unrelated_path', 'unrelated_hash', 'source_hash_with_remote_path'))
def test_h100_A_loader_rejects_wrong_path_or_hash(tmp_path, monkeypatch, change):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch, remote=True)
    rows = json.loads((target / 'arms.json').read_bytes())
    row = rows[0]
    shard = next((target / 'runs' / row['arm']).glob('server_*/decisions_*.jsonl'))
    decs = [json.loads(line) for line in shard.read_bytes().splitlines()]
    if change == 'unrelated_path':
        decs[0]['argv'][1] += '.unrelated'
        plan_path = target / 'h100_sync/plan.json'
        plan = json.loads(plan_path.read_bytes())
        plan['files'] = [dict(original=flag(row['plugin_args'], '--os-fit-artifact'), rel=decs[0]['argv'][1])]
        dump(plan_path, plan)
        message = 'Wrong served artifact'
    else:
        value = b.sha(b'unrelated fit') if change == 'unrelated_hash' else b.sha(Path(flag(row['plugin_args'], '--os-fit-artifact')).read_bytes())
        for d in decs[1:]:
            d['r14']['fit_sha256'] = value
        message = 'Wrong served fit hash' if change == 'unrelated_hash' else 'path/hash branch mismatch'
    jl(shard, decs)
    with pytest.raises(ValueError, match=message):
        a.load_a(cell, levels, uniform, p, caps, a.Reader(), template)


def test_A_loader_rejects_another_roots_remote_run(tmp_path, monkeypatch):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch, remote=True)
    shared = Path(p['test_references']['pi05_l10']['root'])
    path = shared / 'h100_sync/plan.json'
    plan = json.loads(path.read_bytes())
    plan['remote_run'] = '/custom/h100/runs/unrelated_root'
    dump(path, plan)
    with pytest.raises(ValueError, match='Invalid H100 remote run path'):
        a.load_a(cell, levels, uniform, p, caps, a.Reader(), template)


def test_A_hash_snapshot_does_not_read_outcomes_and_defers_running_tail(tmp_path, monkeypatch):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch)
    row = json.loads((target / 'arms.json').read_bytes())[0]
    source = b.sha(Path(flag(row['plugin_args'], '--os-fit-artifact')).read_bytes())
    verifier = a.AServedFitVerifier(target, row, source, a.Reader())
    path = tmp_path / 'snapshot.jsonl'
    tail = b'{"ev":"dec","r14":'
    path.write_bytes(json.dumps(dict(ev='startup', argv=row['plugin_args'])).encode() + b'\n' +
                     json.dumps(dict(ev='dec', r14=dict(fit_sha256=source))).encode() + b'\n' + tail)
    snapshot = decision_paths_hashes(path, verifier)
    assert snapshot['startup_paths'] == snapshot['decisions_with_fit_hash'] == 1
    assert snapshot['incomplete_tail_bytes'] == len(tail)
    assert set(verifier.audit()['observed_path_branches'].values()) == {'frozen_source'}


def test_full_A_hash_audit_blocks_all_outcome_reads(tmp_path, monkeypatch):
    cell, levels, uniform, p, caps, template, target = fake_a(tmp_path, monkeypatch)
    dump(tmp_path / 'cap_tables.json', caps)

    class MetadataOnlyReader(a.Reader):
        def read(self, path):
            path = Path(path)
            assert path.name not in ('summary.json', 'journal.jsonl', 'analysis.json', 'test_a.json')
            assert 'client' not in path.parts
            return super().read(path)

    monkeypatch.setattr(audit, 'TRACE', tmp_path)
    monkeypatch.setattr(audit, 'HERE', tmp_path)
    monkeypatch.setattr(audit, 'CELLS', (cell,))
    monkeypatch.setattr(audit, 'Reader', MetadataOnlyReader)
    monkeypatch.setattr(audit, 'verify_protocol', lambda reader: p)
    result = audit.verify(('pi05',))
    assert result['PASS'] and result['arms_with_verified_paths_and_hashes'] == result['arms'] == 3
    assert not result['pending_observations']
    assert not result['journals_read'] and not result['summaries_read'] and not result['outcome_tables_read']
    assert not result['apply_run']


def test_only_served_fit_checks_and_audit_changed_in_A_loader():
    def loader(path):
        return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'load_a')
    before = loader(a.HERE / 'analyze_test_a.before_served_fit_fix.py.txt')
    after = loader(Path(a.__file__))

    class Restore(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'fit_verifier' for t in node.targets):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            call = node.value
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and \
                    isinstance(call.func.value, ast.Name) and call.func.value.id == 'fit_verifier':
                text = "require(Path(args[args.index('--os-fit-artifact') + 1]) == fit, 'Wrong served artifact')" if call.func.attr == 'check_path' else "require(diag['fit_sha256'] == fit_digest, 'Served fit hash mismatch')"
                return ast.parse(text).body[0]
            return self.generic_visit(node)

        def visit_Call(self, node):
            node.keywords = [k for k in node.keywords if k.arg != 'fit_hash_verification']
            return self.generic_visit(node)

    assert ast.dump(Restore().visit(after), include_attributes=False) == ast.dump(before, include_attributes=False)
