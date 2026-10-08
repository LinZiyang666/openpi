"""CPU tests for H2/M2, real B-library off equivalence and all pilot contracts."""
from __future__ import annotations

import json
from pathlib import Path
import pickle
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

from exp.offline_search.closed_loop.guard import R12Guard, feature_matrix, h2_flags
from exp.offline_search.closed_loop.lookhalf import NeedCamera, _View
from exp.offline_search.closed_loop.lookhalf_cameras import ledger
from exp.offline_search.closed_loop.lookhalf_ledger import summarize
from exp.offline_search.closed_loop.plugin import clone_method, parse_cli
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g3_recovery.g3_core import PseudoQuery, episode_rows
from exp.offline_search.rounds.r10.recipe.fitting import _Library, library_boundary
from .build import HERE, R12, STORE, CELLS

EVIDENCE = {}


def _load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)['method']


def _equal(test, left, right):
    for name in ('topk', 'scores', 'action'):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    test.assertEqual(left.confidence, right.confidence)
    test.assertEqual(left.extras, right.extras)
    test.assertEqual(left.library, right.library)


def _fixture(method='m2', knob=False):
    cell = 'pi05_l10_50'+('_periodic_ir32' if knob else '')
    m = _load(HERE/'artifacts'/cell/f'{method}.pkl')
    lib = _Library(STORE, 'pi05_l10', m.library, m.episode_subset)
    rows = next(r for r in episode_rows(m.inner.C).values() if len(r) > 12)
    queries = {}
    for step in range(0, 12, 2):
        q = PseudoQuery(lib, m.inner.C, rows, step, 'pi05', True)
        ep = q.episode
        queries[step] = _View(q, hist_has_vision=np.arange(step)%2 == 0,
                             episode=api.EpisodeView(ep.uid, ep.task, ep.task_id, 0, ep.index, ep.seed))
    m.reset(queries[0].episode)
    return m, queries


class GuardTests(unittest.TestCase):
    """Guard, camera, default-equivalence and B-pilot invariants."""
    def test_off_equivalence_in_every_cell_and_periodic(self):
        rows_out = []
        cells = [(*cell, 'off') for cell in CELLS]+[('pi05', 'l10', 50, 'periodic')]
        for model, suite, size, knob in cells:
            key = f'{model}_{suite}_{size}'
            source = (R12.parent/'r10/recipe/artifacts'/f'r10_recipe_{key}.pkl' if knob == 'off' else
                      R12.parent/'r11/knob/artifacts'/f'r11_{key}_periodic_ir32.pkl')
            base = _load(source)
            off = _load(HERE/'artifacts'/(key+('_periodic_ir32' if knob == 'periodic' else ''))/'off.pkl')
            lib = _Library(STORE, f'{model}_{suite}', base.library, base.episode_subset)
            episodes = list(episode_rows(base.inner.C).values())
            count = 0
            for task in range(10):
                rows = next(r for r in episodes if base.inner.C.task[r[0]] == task)
                q0 = PseudoQuery(lib, base.inner.C, rows, 0, model, True)
                base.reset(q0.episode); off.reset(q0.episode)
                for step in range(min(12, len(rows))):
                    ep = q0.episode
                    q = _View(PseudoQuery(lib, base.inner.C, rows, step, model, True),
                              hist_has_vision=np.ones(step, bool),
                              episode=api.EpisodeView(ep.uid, ep.task, ep.task_id, 0, ep.index, ep.seed))
                    _equal(self, base.query(q), off.query(q))
                    self.assertEqual(base.inner._vision_progress, off.inner._vision_progress)
                    np.testing.assert_array_equal(base.inner.base._anchor['action'], off.inner.base._anchor['action'])
                    count += 1
            rows_out.append(dict(cell=key, knob=knob, decisions=count, bit_exact=True))
        EVIDENCE['off_equivalence'] = rows_out

    def test_active_full_cache_unchanged_in_every_cell(self):
        count = 0
        for model, suite, size in CELLS:
            key = f'{model}_{suite}_{size}'
            base = _load(R12.parent/'r10/recipe/artifacts'/f'r10_recipe_{key}.pkl')
            active = [_load(HERE/'artifacts'/key/f'{name}.pkl') for name in ('h2', 'm2')]
            lib = _Library(STORE, f'{model}_{suite}', base.library, base.episode_subset)
            rows = next(iter(episode_rows(base.inner.C).values()))
            q0 = PseudoQuery(lib, base.inner.C, rows, 0, model, True)
            for method in [base]+active:
                method.reset(q0.episode)
            for step in range(min(12, len(rows))):
                q = _View(PseudoQuery(lib, base.inner.C, rows, step, model, True),
                          hist_has_vision=np.ones(step, bool))
                expected = base.query(q)
                for method in active:
                    method.set_camera_mode('full')
                    actual = method.query(q)
                    for name in ('topk', 'scores', 'action'):
                        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
                    self.assertEqual(actual.confidence, expected.confidence)
                    self.assertEqual(base.inner._vision_progress, method.inner._vision_progress)
                    self.assertEqual(actual.extras['os_h2_call'], method.lookhalf_diagnostics['h2']['guard_call'])
                    count += 1
        EVIDENCE['active_full_cache'] = dict(decisions=count, rows_scores_actions_corrector_unchanged=True)

    def test_h2_rule_boundaries_and_feature_transform(self):
        np.testing.assert_array_equal(h2_flags([3, 4, 4, 4, 4], [.5, .5, .6, .5, np.nan],
            [.9, .2, .2, .19, .8], .2, .8), [False, True, False, False, True])
        table = dict(W_d1rel=[1., 2.], step=[0, 100], lagrel=[0., 1.], speed2=[np.nan, .1],
                     osc=[0., 2.], mismatch=[np.nan, 3.], W_adv=[np.nan, 12.])
        X = feature_matrix(table)
        np.testing.assert_array_equal(X[:, 1], [0, 1])
        np.testing.assert_array_equal(X[:, -1], [5, 8])
        np.testing.assert_allclose(X[:, 5], np.log([2, 4]))

    def test_h2_replaces_verdict_before_knob_and_old_blind_veto(self):
        m, q = _fixture('h2', True)
        m.query(q[0]); m.query(q[2])
        pending = m._prepare_guard(q[4])
        pending['guard_call'] = True
        pending['scalars']['call'] = True
        before = len(m.inner._vision_progress)
        result = m.query(q[4])
        self.assertEqual(result.extras['os_reason'], 4.)
        self.assertEqual(result.extras['os_r11_guard'], 1.)
        self.assertEqual(result.extras['os_r11_knob_call'], 0.)
        self.assertEqual(len(m.inner._vision_progress), before+1)
        _equal(self, result, m.query(q[4]))
        self.assertEqual(len(m.inner._vision_progress), before+1)
        m.inner.deployed._noprog_span = 100
        marker = object()
        with patch.object(m.inner.base, 'blind_step', return_value=marker):
            self.assertIs(m.inner.blind_step(NS()), marker)
        EVIDENCE['h2_verdict'] = dict(before_knob=True, old_blind_veto_removed=True, retries_idempotent=True)

    def test_completion_is_pure_and_h2_decision_stands(self):
        m, q = _fixture()
        m.query(q[0]); m.query(q[2])
        other, _ = clone_method(m, strict=True)
        m.set_camera_mode('wrist_only')
        pending = m._prepare_guard(q[4]); pending['guard_call'] = True
        before = pickle.dumps(m.inner.deployed)
        result = m.query(_View(q[4], key_v0=np.full(32768, np.nan, np.float32)))
        self.assertEqual(result, NeedCamera(reason='h2_call'))
        self.assertEqual(before, pickle.dumps(m.inner.deployed))
        self.assertEqual(m._lh_step, 2)
        self.assertIsNone(m.inner.decision)
        m.set_camera_mode('full')
        result = m.query(q[4])
        self.assertEqual(result.extras['os_reason'], 4.)
        self.assertEqual(result.extras['os_h2_risk'], pending['scalars']['risk'])
        self.assertEqual(other._lh_step, 2)
        self.assertIsNone(other._pending_guard)
        EVIDENCE['completion'] = dict(rejection_no_guard_commit=True, frozen_h2=True, session_isolation=True)

    def test_m2_accepts_wl_head_tail_without_hidden_camera(self):
        m, q = _fixture()
        m.query(q[0]); m.query(q[2])
        m.risk_fit = dict(m.risk_fit, theta_a=1., theta_b=1.)
        m.set_camera_mode('wrist_only')
        pending = m._prepare_guard(q[4])
        W = pending['W']
        proposal = dict(W, raw=W['raw'].copy())
        original = m._query_metric
        def retrieve(name, *args):
            return dict(proposal) if name.startswith('opus_l') else original(name, *args)
        poison = _View(q[4], key_v0=np.full(32768, np.nan, np.float32))
        with patch.object(m, '_query_metric', side_effect=retrieve):
            prepared = m._m2(poison, pending)
            result = m.query(poison)
        self.assertFalse(result.extras['os_force_miss'])
        np.testing.assert_array_equal(result.action, prepared['action'])
        np.testing.assert_array_equal(result.action[:, 6:], W['raw'][:, 6:])
        np.testing.assert_array_equal(m.inner.base._anchor['action'], result.action)
        self.assertEqual(m._memory['step'], 2)
        self.assertEqual(len(m._guard_history), 3)
        self.assertTrue(m.lookhalf_diagnostics['accepted_half'])
        self.assertFalse(m.half_fit['metrics']['opus_w'][0][0].Z.flags.writeable)
        EVIDENCE['m2_synthesis'] = dict(WL_payload=True, wrist_head=True, hidden_camera_poisoned=True,
                                      authoritative_tail=True, gripper_padding_unchanged=True)

    def test_m2_lag_consistency_and_knob_completion(self):
        m, q = _fixture()
        m.query(q[0])
        m.risk_fit = dict(m.risk_fit, theta_a=1., theta_b=1.)
        m.set_camera_mode('wrist_only')
        pending = m._prepare_guard(q[10])
        self.assertEqual(m._m2(q[10], pending), NeedCamera(reason='lag'))
        m._pending_guard = None
        pending = m._prepare_guard(q[2])
        original = m._query_metric
        def disagree(name, *args):
            result = original(name, *args)
            result['raw'] = result['raw'].copy()+100
            return result
        with patch.object(m, '_query_metric', side_effect=disagree):
            self.assertEqual(m._m2(q[2], pending), NeedCamera(reason='consistency'))
        with patch.object(m, 'periodic_due', return_value=True):
            self.assertEqual(m._m2(q[2], pending), NeedCamera(reason='knob_call'))

    def test_m2_uses_call_memory_and_lifecycle(self):
        m, q = _fixture()
        m.query(q[0])
        m._memory['called'] = True
        m._previous_look_called = True
        metadata = dict(look_reason=6, executed_steps=5, execution_audit=True, exec_ok=True)
        self.assertEqual(m.plan_cameras(q[2], **metadata), 'wrist_only')
        with patch.object(m, 'periodic_due', return_value=True):
            self.assertEqual(m.plan_cameras(q[2], **metadata), 'full')
        self.assertEqual(m.plan_cameras(q[2], **dict(metadata, execution_audit=False)), 'full')
        self.assertIsNone(m._memory)

    def test_fifteen_b_roots_and_plugin_options(self):
        from exp.offline_search.closed_loop import devset
        from exp.offline_search.closed_loop.look_recording import validate_root
        from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
        records = json.loads((HERE/'pilot_roots.json').read_text())
        self.assertEqual(len(records), 5)
        total = 0
        for record in records:
            root = Path(record['root']); validate_root(root)
            self.assertEqual((root/'dev_selection.json').read_bytes(), Path(record['source_selection']).read_bytes())
            selected = load_manifest(root/'dev_selection.json')
            self.assertEqual(len(selected['selected']), 100)
            for row in json.loads((root/'arms.json').read_text()):
                contract = devset.arm_contract(root, row, selected)
                self.assertEqual(contract['init_pool'], 'B')
                path = HERE/'scratch/contracts'/f"{row['arm']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(contract, schema='offline_search.dev_looks.v1', cell=row['cell'],
                    method=row['method'], recording_arm_dir=str(root/'runs'/row['arm']), pairs=sorted(selected['selected'])))+'\n')
                opts, rest = parse_cli(['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']),
                    '--os-cell', f"{row['model']}_{row['suite_short']}_cache", '--os-log-dir', str(root/'runs'/row['arm']),
                    '--os-look-contract', str(path), '--os-look-contract-sha256', devset.sha(path)]+row['plugin_args'])
                self.assertFalse(rest)
                self.assertEqual(opts.os_request_cameras, row['kwargs']['method'] == 'm2')
                self.assertTrue(opts.os_record_looks)
                self.assertEqual(row['kwargs']['knob_method'], record['knob'])
                self.assertEqual(row['server_host'], 'local')
                total += 1
        self.assertEqual(total, 15)
        EVIDENCE['pilots'] = dict(roots=5, arms=15, episodes_per_arm=100, K=10, B_disjoint=True, launches=0)

    def test_compute_ledger(self):
        for model, half in (('pi05', .076), ('groot', .074)):
            rows = []
            for step, vision, wrist, completion, miss in ((0, True, False, 0, True),
                    (1, False, False, 0, False), (2, True, True, 0, False),
                    (3, True, True, 1, True), (4, True, True, 1, False)):
                rows.append(dict(uid='B', step=step, vision=vision, hit=not miss, extras={},
                    lookhalf_cost=ledger(model, vision=vision, wrist_first=wrist, completions=completion, miss=miss)))
            s = summarize(model, rows)
            self.assertAlmostEqual(s['work'], 2+3*half)
            self.assertEqual(s['one_camera_looks'], 1)
            self.assertEqual(s['encoded_cameras'], 7)
            with self.assertRaises(ValueError):
                ledger(model, vision=True, wrist_first=True, completions=0, miss=True)
        EVIDENCE['ledger'] = dict(full_calls_cost_one=True, completion_once=True, blind_zero=True,
                                  prices={'pi05': .076, 'groot': .074})

    def test_recording_traverses_h2_and_keeps_features(self):
        from exp.offline_search.closed_loop.look_recording import retrieval_snapshot
        m, q = _fixture('h2')
        result = m.query(q[0])
        snap = retrieval_snapshot(m, result.topk, result.scores, result.extras)
        np.testing.assert_array_equal(snap['donor_rows'], result.topk)
        self.assertTrue(np.isfinite(snap['donor_weights']).all())
        self.assertTrue(any('_vision_progress' in state for state in snap['guard_state']['method'].values()))
        self.assertEqual(len(m.lookhalf_diagnostics['h2']['features']), 7)
        self.assertIn('os_h2_risk', snap['guard_state']['extras'])
        EVIDENCE['recording'] = dict(donor_rows_weights=True, guard_memos=True, seven_features=True, risk_and_branches=True)

    def test_coordinator_environment_and_reserved_ports(self):
        script = HERE/'coordinator_commands.sh'
        subprocess.run(['bash', '-n', str(script)], check=True, capture_output=True)
        for ports, count in [('23100,23101,23102,23103', '4'), ('23120', '4')]:
            env = dict(os.environ, PORTS=ports, LOCAL_SERVER_COUNT=count)
            result = subprocess.run(['bash', str(script), 'report'], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
        env = dict(os.environ, PORTS='23144', LOCAL_SERVER_COUNT='1')
        with tempfile.TemporaryDirectory(prefix='r12_guard_report_test_') as scratch:
            base = Path(scratch) / 'pilots'
            for record in json.loads((HERE / 'pilot_roots.json').read_text()):
                source = Path(record['root'])
                target = base / source.name
                target.mkdir(parents=True)
                (target / 'arms.json').write_bytes((source / 'arms.json').read_bytes())
            isolated = Path(scratch) / 'coordinator_commands.sh'
            isolated.write_text(script.read_text().replace(f'BASE={HERE / "pilots"}', f'BASE={base}'))
            subprocess.run(['bash', str(isolated), 'report'], env=env, capture_output=True, check=True)
        preflight = script.read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        listening = 'header\n0: 0100007F:5A3C 00000000:0000 0A\n'
        for action in ('sync', 'chain'):
            with patch.dict(os.environ, PORTS='23120', LOCAL_SERVER_COUNT='1'), \
                    patch.object(sys, 'argv', ['preflight', action]), \
                    patch.object(Path, 'read_text', return_value=listening):
                with self.assertRaisesRegex(SystemExit, 'look-half pilot is still listening'):
                    exec(compile(preflight, str(script), 'exec'), {})
        EVIDENCE['coordinator'] = dict(default_servers=4, honors_env=True, reserved_ports_refused=True,
                                      port_count_mismatch_refused=True, running_pilot_refused=['sync', 'chain'],
                                      actions_tested=['report', 'isolated preflight'], launches=0)

    def test_library_boundary_refuses_a_and_dev(self):
        with library_boundary(STORE, STORE/'library/pi05_l10/bpool_cs'):
            with self.assertRaises(ValueError):
                (STORE/'queries/pi05_l10_cache/meta.json').open('rb')
            with self.assertRaises(ValueError):
                Path('/home/weiland/trace_runs/os_closed_loop/r12_devlooks_50L/arms.json').open('rb')


def main():
    """Write reproducible CPU test evidence and return failure on any error."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    EVIDENCE['tests'] = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors), cpu_only=True)
    (HERE/'SELFTEST.json').write_text(json.dumps(EVIDENCE, indent=2)+'\n')
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
