"""Real CPU plugin replay; frozen corr3 artifacts are verification inputs only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import pickle
import time
from types import MethodType, SimpleNamespace

import numpy as np

from exp.offline_search.rounds.r10.data import STORE, RUNS, CELLS, SIZES, PARENT, sha, write_json
from .build import HERE, SPEC, CURRENT, artifact_path, run_cpu


def reference_rows(model, suite, size):
    name = f'r10_{model}_{suite}_{size}_GC_dist'
    rows = []
    for suffix in ('', '_b'):
        root = RUNS / f'r10_corr3_{model}{suffix}'
        if not (root / 'arms.json').exists():
            continue
        for row in json.loads((root / 'arms.json').read_text()):
            if row['arm'] == name and Path(row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1]).exists():
                rows.append((root, row))
    if not rows:
        raise ValueError(f'no frozen corr3 fit for {name}')
    return rows


def recipe_row(reference, model, suite, size):
    path = artifact_path(model, suite, size)
    with path.open('rb') as f:
        blob = pickle.load(f)
    row = dict(reference)
    row.update(arm=path.stem, method=SPEC, kwargs=blob['kwargs'])
    flags = list(reference['plugin_args'])
    flags[flags.index('--os-fit-artifact') + 1] = str(path)
    row['plugin_args'] = flags
    return row


def replay(row, label, size=None):
    from exp.offline_search.closed_loop import plugin
    from exp.offline_search.closed_loop.selftest import FakeKB, FakePolicy
    from exp.offline_search.harness import store
    from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
    import openpi.cache.config as cc
    from openpi.cache.orchestrator import CacheOrchestrator

    plugin._git_head = lambda: None
    output = HERE / 'replays' / label
    # Remove only this replay's own synthetic decision files, never real logs.
    for p in output.glob('decisions_*.jsonl'):
        p.unlink()
    args = ['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']), '--os-cell', row['cell'],
            '--os-log-dir', str(output), '--os-tag', 'recipe_replay', *row['plugin_args']]
    opts, rest = plugin.parse_cli(args)
    assert not rest
    rt = plugin.install(opts, row['model'])
    cfg = cc.load_cache_config(row['yaml'])
    shared = cc.build_shared_storage(cfg)
    library = 'current' if size is None else PARENT[row['model']]
    lib = store.LibraryView(STORE, f'{row["model"]}_{row["suite_short"]}', library)
    qc = SimpleNamespace(key_v0=lib.key_v0, key_v1=lib.key_v1, rs=lib.rs)

    class CPUFake(FakePolicy):
        def stage1(self, obs):
            pass

        def _osp_prepare_blind(self, obs):
            return np.array(lib.rs[obs['_row']], copy=True), None

        def _osp_blind_output(self, action, state):
            return {'actions': action.copy()}

    def factory(_base, bundle_id='default'):
        comps = cc.build_per_connection_components(cfg, shared, quiet=True)
        kb = FakeKB(qc)
        for s in plugin._TLS.new_sessions:
            s.kb = kb
        orch = CacheOrchestrator(storage=comps['storage'], key_builder=kb, gates=comps['gates'],
            judges=comps['judges'], search_strategies=comps['search_strategies'], timer=comps['timer'],
            write_policy=comps.get('write_policy'), offline_writers=comps.get('offline_writers', ()),
            library_stats=comps.get('library_stats'))
        return CPUFake(orch, kb, lib.action)

    conns = [plugin._wrap_factory(factory)(None, str(i)) for i in range(2)]
    synth_checks, fingerprints, gate_factors = [], [], []
    for conn in conns:
        inner = conn._osp_sessions[0].method.inner
        progress = inner._progress
        def forced(self, q, top1, original=progress):
            span = original(q, top1)
            if int(q.step) in (2, 8):
                self._noprog_span = self.noprog_n - 1
                return self._noprog_span
            return span
        inner._progress = MethodType(forced, inner)  # never serialized in artifacts
        base = inner.base
        synth = base.os_synth
        def checked(self, q, rows, weights, original=synth):
            raw = BlindAWM.os_synth(self, q, rows, weights)
            correction = self._correction(q, raw)
            expected = self._apply(raw, correction)
            action = original(q, rows, weights)
            assert action.tobytes() == expected.tobytes()
            assert action[:, 6:].tobytes() == raw[:, 6:].tobytes()
            assert action[10:].tobytes() == raw[10:].tobytes()
            if int(q.step) == 0:
                assert action.tobytes() == raw.tobytes()
            T, _, _, _, _, _, _, d, _, _, _ = self._dist(q)
            positions = np.searchsorted(T.rows, rows)
            ratio = float(np.min(d[positions])) / self.distance_scale
            gate_factors.append(0. if q.step == 0 else float(np.clip((2-ratio)/1.25, 0, 1)))
            fingerprints.append(dict(step=int(q.step), task=int(q.task_id), rows=np.asarray(rows).tobytes().hex(),
                weights=np.asarray(weights).tobytes().hex(), raw=raw.tobytes().hex(), action=action.tobytes().hex()))
            synth_checks.append(1)
            return action
        base.os_synth = MethodType(checked, base)

    actions = []
    # Ten tasks, two simultaneous sessions, each reset three times: static
    # no-progress states, sequential library states, then distant library keys.
    for pair in range(5):
        for scenario in range(3):
            for i, conn in enumerate(conns):
                t = 2 * pair + i
                conn.on_episode_start(task=lib.meta['tasks'][str(t)], episode_id=scenario,
                    extra_metadata={'task_uid': f'replay-{t}-{scenario}', 'task_id': t, 'orig_init_state_idx': scenario})
            for step in range(12):
                for i, conn in enumerate(conns):
                    t = 2 * pair + i
                    candidates = np.flatnonzero(lib.task_id == t)
                    if scenario == 0:
                        r = int(candidates[0])
                    elif scenario == 1:
                        r = int(candidates[min(step, len(candidates)-1)])
                    else:
                        r = int(candidates[(step * 97 + len(candidates)//2) % len(candidates)])
                    obs = {'observation/state': np.asarray(lib.rs[r, :8], np.float64),
                           'prompt': lib.meta['tasks'][str(t)], '_row': r,
                           'observation/image': np.zeros((2, 2, 3), np.uint8),
                           'observation/wrist_image': np.zeros((2, 2, 3), np.uint8),
                           '__extra__': {'decision_id': step, 'executed_steps': 5}}
                    result = conn.infer(obs)
                    sess = conn._osp_sessions[0]
                    assert sess.step == step + 1 == sess.b_aex.n
                    assert np.array_equal(sess.b_aex.a[step], result['actions'])
                    assert np.isfinite(result['actions'][:, :7]).all()
                    actions.append(np.asarray(result['actions']).tobytes().hex())
            for conn in conns:
                conn.on_episode_end(success=False)
    rt.flush_all()
    decs = [json.loads(line) for line in rt.dec_path.read_text().splitlines() if '"ev": "dec"' in line]
    assert len(decs) == 360
    assert all(d['lib'] == library for d in decs if d['src'] in ('cache', 'cache_blind'))
    forced_decs = [d for d in decs if d['step'] in (2, 8)]
    assert len(forced_decs) == 60
    assert all(d['vision'] and not d['hit'] and d['extras']['os_force_miss'] == 1
               and int(d['extras']['os_flags']) == 8 for d in forced_decs)
    assert any(d['src'] == 'policy_tail' for d in decs)
    assert any(d['src'] == 'cache_blind' for d in decs)
    # Compare all decision semantics. Machine/time/clone identifiers do not
    # describe a decision; preserve scores, confidence, rows, flags and extras.
    ignored = {'ts', 'conn', 'fit_s', 'method', 'tag', 'wall_ms', 'timing', 'timings', 'ms'}
    semantic = [{k: v for k, v in d.items() if k not in ignored and not k.endswith(('_ms', '_us'))} for d in decs]
    report = dict(PASS=True, decisions=len(decs), tasks=10, connections=2, episodes=30,
                  forced_guard_triggers=len(forced_decs), serving_corrector_checks=len(synth_checks),
                  miss=sum(not d['hit'] for d in decs), blind=sum(d['src'] == 'cache_blind' for d in decs),
                  policy_tail=sum(d['src'] == 'policy_tail' for d in decs),
                  gate_zero=sum(f == 0 for f in gate_factors),
                  gate_partial=sum(0 < f < 1 for f in gate_factors), gate_full=sum(f == 1 for f in gate_factors))
    write_json(output / 'selftest_report.json', report)
    return dict(decisions=semantic, actions=actions, synthesis=fingerprints), report


def fitted_check(new_path, old_path):
    with Path(new_path).open('rb') as f:
        new = pickle.load(f)['method']
    with Path(old_path).open('rb') as f:
        old = pickle.load(f)['method']
    nb, ob = new.inner.base, old.inner.base
    np.testing.assert_array_equal(new.row_subset, old.row_subset)
    for key in ('sig', 'mu0', 'mu1', 'muB0', 'muB1', 'B0T', 'B1T', 'lib_ep', 'lib_step'):
        assert getattr(nb, key).tobytes() == getattr(ob, key).tobytes(), key
    for task in ob.tasks:
        for key, value in vars(ob.tasks[task]).items():
            if isinstance(value, np.ndarray):
                assert value.tobytes() == getattr(nb.tasks[task], key).tobytes(), (task, key)
            else:
                assert value == getattr(nb.tasks[task], key), (task, key)
    for task in ob.heads:
        for key, value in ob.heads[task].items():
            assert value.tobytes() == nb.heads[task][key].tobytes(), (task, key)
    assert nb.distance_scale == ob.distance_scale
    assert nb.distance_rule == ob.distance_rule
    assert tuple(new.inner.disabled_guards) == tuple(old.inner.disabled_guards)
    def equal(a, b):
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        elif isinstance(a, dict):
            assert set(a) == set(b)
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for aa, bb in zip(a, b):
                equal(aa, bb)
        else:
            assert a == b or (isinstance(a, float) and np.isnan(a) and np.isnan(b))
    for key in ('sigma', 'm_thr', 'M0', 'M1', 'c_thr', 'med_len', 'thr_stats', 'cal', 'disp_thr', 'disp_qgrid'):
        equal(getattr(new.inner, key), getattr(old.inner, key))
    for key, value in vars(old.inner.C).items():
        if key not in ('act', 'name'):
            equal(getattr(new.inner.C, key), value)
    return dict(cache_arrays_byte_identical=True, head_arrays_byte_identical=True,
                guard_calibration_byte_identical=True, distance_scale_identical=True, distance_scale=nb.distance_scale)


def isolated_replay(row, label, size=None):
    rowfile = HERE / 'replays' / label / 'row.json'
    write_json(rowfile, row)
    args = ['-m', 'exp.offline_search.rounds.r10.recipe.replay', 'trace',
            '--row', str(rowfile), '--label', label]
    if size is not None:
        args += ['--size', str(size)]
    run_cpu(args, HERE / 'validation' / f'{label}_trace.log')
    with gzip.open(rowfile.parent / 'trace.pkl.gz', 'rb') as f:
        return pickle.load(f)


def compare_one(model, suite, size):
    references = reference_rows(model, suite, size)
    _, first = references[0]
    newrow = recipe_row(first, model, suite, size)
    newtrace, newreport = isolated_replay(newrow, f'{model}_{suite}_{size}_recipe', size)
    comparisons = []
    for root, row in references:
        oldpath = row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1]
        fitted = fitted_check(artifact_path(model, suite, size), oldpath)
        trace, report = isolated_replay(row, f'{model}_{suite}_{size}_{root.name}', size)
        # Write reviewable differences before failing, without accepting tolerance.
        diffs = {key: [i for i, (a, b) in enumerate(zip(newtrace[key], trace[key])) if a != b]
                 for key in newtrace}
        write_json(HERE / 'equivalence' / f'{model}_{suite}_{size}_{root.name}.json',
                   dict(reference=str(root), differences=diffs, recipe=newreport, frozen=report))
        assert all(len(newtrace[k]) == len(trace[k]) for k in newtrace)
        if any(diffs.values()):
            # First semantic difference, compact enough to diagnose a failed replay.
            for k, ids in diffs.items():
                if ids:
                    print('DIFFERENCE', k, ids[0], newtrace[k][ids[0]], trace[k][ids[0]], flush=True)
                    break
            raise AssertionError(f'replay differences: {model}_{suite}_{size}: {diffs}')
        comparisons.append(dict(reference=str(root), artifact_sha256=sha(oldpath), differences=0,
                                decisions=report['decisions'], forced_guard_triggers=report['forced_guard_triggers'], **fitted))
    report = dict(PASS=True, model=model, suite=suite, size=size, artifact=str(artifact_path(model, suite, size)),
                  artifact_sha256=sha(artifact_path(model, suite, size)), recipe=newreport, comparisons=comparisons)
    write_json(HERE / 'equivalence' / f'{model}_{suite}_{size}.json', report)
    print(json.dumps(report), flush=True)


def current_one(model, suite):
    row, = [r for r in json.loads((CURRENT / 'arms.json').read_text())
             if r['model'] == model and r['suite_short'] == suite]
    _, report = replay(row, f'{model}_{suite}_current')
    report.update(model=model, suite=suite, artifact_sha256=sha(artifact_path(model, suite)))
    write_json(HERE / 'current_selftests' / f'{model}_{suite}.json', report)
    print(json.dumps(report), flush=True)


def validate(workers, wait_build=False):
    jobs = [(m, s, n) for m, s in CELLS for n in SIZES] + [(m, s, None) for m, s in CELLS]
    def one(job):
        model, suite, size = job
        output = artifact_path(model, suite, size)
        if wait_build:
            deadline = time.monotonic() + 3600
            while not output.exists() or (size is None and not (CURRENT / 'arms.json').exists()):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'awaited recipe output: {output}')
                time.sleep(1)
        if size is not None:
            report_path = HERE / 'equivalence' / f'{model}_{suite}_{size}.json'
            if report_path.exists():
                prior = json.loads(report_path.read_text())
                # Preserve completed comparisons only while both artifacts match.
                refs = reference_rows(model, suite, size)
                expected = [(str(root), sha(row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1]))
                            for root, row in refs]
                got = [(c['reference'], c['artifact_sha256']) for c in prior['comparisons']]
                if (prior['PASS'] and prior['artifact_sha256'] == sha(output) and got == expected
                        and all(c.get('guard_calibration_byte_identical') for c in prior['comparisons'])):
                    print(f'REPLAY READY {model}_{suite}_{size}', flush=True)
                    return
        args = ['-m', 'exp.offline_search.rounds.r10.recipe.replay', 'one', '--model', model, '--suite', suite]
        if size is not None:
            args += ['--size', str(size)]
        name = f'{model}_{suite}_{size or "current"}'
        run_cpu(args, HERE / 'validation' / f'{name}_replay.log')
        print(f'REPLAY {name}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, jobs))
    records = [json.loads((HERE / 'equivalence' / f'{m}_{s}_{n}.json').read_text()) for m, s in CELLS for n in SIZES]
    write_json(HERE / 'equivalence.json', dict(PASS=True, cell_sizes=24,
        compared_reference_artifacts=sum(len(r['comparisons']) for r in records),
        differing_decisions=0, compared_decisions=sum(c['decisions'] for r in records for c in r['comparisons']),
        forced_guard_triggers=sum(c['forced_guard_triggers'] for r in records for c in r['comparisons']), records=records))
    current = [json.loads((HERE / 'current_selftests' / f'{m}_{s}.json').read_text()) for m, s in CELLS]
    write_json(HERE / 'current_selftests.json', current)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['one', 'all', 'trace'])
    p.add_argument('--model', choices=['pi05', 'groot'])
    p.add_argument('--suite', choices=['l10', 'spatial'])
    p.add_argument('--size', type=int, choices=SIZES)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--wait-build', action='store_true', help='wait for this recipe builder, never a remote evaluation')
    p.add_argument('--row', type=Path)
    p.add_argument('--label')
    a = p.parse_args()
    if a.action == 'trace':
        if a.row is None or a.label is None:
            p.error('trace needs --row and --label')
        result = replay(json.loads(a.row.read_text()), a.label, a.size)
        with gzip.open(HERE / 'replays' / a.label / 'trace.pkl.gz', 'wb') as f:
            pickle.dump(result, f, protocol=4)
    elif a.action == 'all':
        validate(a.workers, a.wait_build)
    elif not a.model or not a.suite:
        p.error('one needs --model and --suite')
    elif a.size is None:
        current_one(a.model, a.suite)
    else:
        compare_one(a.model, a.suite, a.size)


if __name__ == '__main__':
    main()
