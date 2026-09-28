"""Installed plugin blind driver on all deployed fits and exact inherited controls."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from prepare import PREFIX, ROOT

HERE = Path(__file__).resolve().parent


def worker(args):
    from exp.offline_search.closed_loop import plugin, selftest
    parse = plugin.parse_cli
    plugin.parse_cli = lambda argv: parse([*argv, '--os-no-shadow-native'])
    return selftest.main(args)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    arms = json.loads((HERE / 'arms_q1.json').read_text())
    reports = []
    for arm in arms:
        variants = ('deployed', 'inherited', 'k10') if 'c10' in arm['name'] else ('deployed',)
        comparisons = {}
        for variant in variants:
            tag = arm['name'] + '_' + variant
            method, kw = arm['method'], dict(arm['kwargs'])
            flags = []
            if variant == 'deployed':
                flags = ['--fit-artifact', f"/tmp/q1_fits/{arm['name']}.pkl"]
            elif variant == 'inherited':
                kw['policy_tail_gate'] = 'inherited'
            else:
                method = 'exp.offline_search.rounds.r04.k10_policy_tail.judge:PolicyTailJudge'
                kw.pop('policy_tail_gate')
                kw.pop('monitor')
            command = PREFIX + [str(HERE / 'plugin_matrix.py'), '--worker', '--blind',
                       '--cell', f"pi05_{arm['suite']}_cache", '--root', ROOT,
                       '--yaml', f"exp/trace_dual/config/tr_pi05_{'sp' if arm['suite']=='spatial' else 'l10'}_cache.yaml",
                       '--method', method, '--kwargs', json.dumps(kw), '--judge', 'guard_only',
                       '--out', str(out / tag)] + flags
            if 'c10' in arm['name']:
                command.append('--policy-tail')
            with (out / (tag + '.log')).open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            report = json.loads((out / tag / 'selftest_report.json').read_text())
            assert report['PASS']
            eligible = tails = 0
            records = []
            for path in sorted((out / tag / 'inputs').glob('*.npz')):
                with np.load(path, allow_pickle=False) as d:
                    if 'source' in d:
                        wanted = np.flatnonzero((d['hit'][:-1] == 0) & d['vision'][:-1]) + 1
                        eligible += len(wanted)
                        used = np.flatnonzero(d['source'] == 'policy_tail')
                        tails += len(used)
                        if variant == 'deployed':
                            assert np.array_equal(wanted, used), (tag, wanted, used)
                            assert np.array_equal(d['a_exec'][used, :5], d['a_exec'][used-1, 5:10])
                    records.append({k: np.array(d[k]) for k in d.files if k != 'meta' and not k.endswith(('_ms', '_us'))})
            if variant in ('inherited', 'k10'):
                comparisons[variant] = records
            reports.append(dict(tag=tag, command=command, eligible_misses=eligible, tails=tails, **report))
            print(tag, report, 'eligible_misses', eligible, flush=True)
            (out / 'summary.json').write_text(json.dumps(reports, indent=2))
        if 'inherited' in comparisons:
            x, y = comparisons['inherited'], comparisons['k10']
            assert len(x) == len(y)
            for a, b in zip(x, y):
                assert a.keys() == b.keys()
                for key in a:
                    if a[key].dtype.kind in 'fc':
                        assert np.array_equal(a[key], b[key], equal_nan=True), key
                    else:
                        assert np.array_equal(a[key], b[key]), key
            reports[-1]['inherited_exact_all_npz_fields'] = True
    (out / 'summary.json').write_text(json.dumps(reports, indent=2))


if __name__ == '__main__':
    import sys
    if '--worker' in sys.argv:
        raise SystemExit(worker([a for a in sys.argv[1:] if a != '--worker']))
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)
