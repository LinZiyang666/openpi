"""Real CPU plugin selftests with LOCAL identity-first readers, standard store.

No serving-store copy and no control/runner adapter. The temporary reader patch
only exists inside this offline test process; emitted server arguments are stock.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .data import HERE, RUN, STORE, episode_metadata, dump


def install_filtered_readers():
    from exp.offline_search.harness import store
    from exp.offline_search.closed_loop import plugin
    original = store.QueryCell
    pairs = {(t, i) for t in (0, 1) for i in range(20)}
    metadata = {}
    for model in ('pi05', 'groot'):
        for suite in ('l10', 'spatial'):
            path = STORE / 'queries' / f'{model}_{suite}_cache' / 'episodes.json'
            candidates = list(episode_metadata(path, pairs))
            # Spatial episodes may finish before 24 rows; pick first long-enough
            # admitted training episode for each of two reset identities.
            eps = [min((e for e in candidates if e['task_id']==t and e['end']-e['start']>=24),
                       key=lambda e:e['init']) for t in (0,1)]
            assert len(eps) == 2
            metadata[path] = eps
    original_read = Path.read_text

    def safe_read(path, *args, **kw):
        if path in metadata:
            return json.dumps(metadata[path])
        if '.git' in path.parts:
            raise AssertionError('git access prohibited')
        return original_read(path, *args, **kw)

    Path.read_text = safe_read  # plugin's identity map must also see only admitted records.
    plugin._git_head = lambda: None

    class FilteredQueryCell(original):
        def __init__(self, root, cell):
            assert Path(root) == STORE
            self.root, self.cell = Path(root), cell
            self.model, self.suite, self.arm = store.parse_cell(cell)
            self.lib_key = f'{self.model}_{self.suite}'
            self.dir = STORE / 'queries' / cell
            self.episodes, self.selected_rows, self._a = [], [], {}
            for e in metadata[self.dir / 'episodes.json']:
                assert 0 <= e['init'] < 20 and e['end'] - e['start'] >= 24
                start = len(self.selected_rows)
                self.selected_rows.extend(range(e['start'], e['start'] + 24))
                self.episodes.append(dict(e, start=start, end=start+24, num_steps=24))
            self.selected_rows = np.asarray(self.selected_rows, np.int64)

        def _arr(self, name):
            if name not in self._a:
                if name == 'ep':
                    self._a[name] = np.repeat(np.arange(2), 24)
                else:
                    src = np.load(self.dir / f'{name}.npy', mmap_mode='r', allow_pickle=False)
                    self._a[name] = np.array(src[self.selected_rows])
            return self._a[name]
    store.QueryCell = FilteredQueryCell
    return metadata


def one(name, forced):
    from exp.offline_search.closed_loop import selftest
    row = next(a for a in json.loads((RUN / 'arms.json').read_text()) if a['arm'] == name)
    kw = dict(row['kwargs'])
    out = RUN / 'selftest' / (name + ('_forced' if forced else '_prod'))
    if out.exists():
        if (out/'admission.json').exists():
            raise ValueError('refuse overwriting successful selftest evidence')
        i=1
        while out.with_name(out.name+f'_failed{i}').exists(): i+=1
        out.rename(out.with_name(out.name+f'_failed{i}'))
    argv = ['--cell', row['cell'], '--yaml', row['yaml'], '--method', row['method'], '--root', str(STORE),
            '--blind', '--policy-tail', '--policy-tail-blocks', '1', '--judge', 'guard_only', '--no-shadow',
            '--out', str(out)]
    if forced:
        kw.update(max_calls=2, force_trigger_at=[2, 6])
    else:
        p = row['plugin_args']
        argv += ['--fit-artifact', p[p.index('--os-fit-artifact')+1]]
    argv += ['--kwargs', json.dumps(kw)]
    metadata = install_filtered_readers()
    rc = selftest.main(argv)
    report = json.loads((out / 'selftest_report.json').read_text())
    assert rc == 0 and report['PASS'] and (not forced or report['miss'] > 0)
    pairs = [[e['task_id'],e['init']] for e in metadata[STORE/'queries'/row['cell']/'episodes.json']]
    dump(out / 'admission.json', dict(pairs=pairs, rows_per_episode=24,
        connections=2, forced=forced, standard_store=str(STORE), reader_patch='CPU selftest process only'))
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm')
    p.add_argument('--forced', action='store_true')
    args = p.parse_args()
    if args.arm:
        return one(args.arm, args.forced)
    results = []
    for row in json.loads((RUN / 'arms.json').read_text()):
        for forced in (False, True):
            name = row['arm']
            tag = name + ('_forced' if forced else '_prod')
            log = RUN / 'selftest' / (tag + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            report = log.parent / tag / 'selftest_report.json'
            admission = report.with_name('admission.json')
            if not (report.exists() and admission.exists()):
                if log.exists():
                    i=1
                    while log.with_name(log.stem+f'_failed{i}.log').exists(): i+=1
                    log.rename(log.with_name(log.stem+f'_failed{i}.log'))
                cmd = [sys.executable, '-m', __package__+'.selftest', '--arm', name]
                if forced:
                    cmd.append('--forced')
                with log.open('w') as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
                if rc:
                    raise SystemExit(f'CPU selftest failed: {log}')
            r = json.loads(report.read_text())
            assert r['PASS']
            results.append(dict(arm=name, forced=forced, report=r))
            print(tag, 'PASS', flush=True)
    dump(HERE / 'results/plugin_selftests.json', results)


if __name__ == '__main__':
    raise SystemExit(main())
