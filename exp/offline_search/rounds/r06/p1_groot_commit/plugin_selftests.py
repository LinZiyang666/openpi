"""Installed plugin blind selftest (closed_loop/selftest.py --blind) on the four deployed GR00T B fits.

Two interleaved connections, two episodes each, stage-1 accounting, duplicate-id rejection, preflight fallbacks,
--os-log-inputs, and the offline verify_logs replay of every logged input through the harness (online == offline).
The worker adds --os-no-shadow-native like the deployed arms (same wrapper as rounds/r05/q1_commit/plugin_matrix.py).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = '/home/weiland/trace_runs/offline_search_store'
FITS = Path('/home/weiland/trace_runs/os_closed_loop/r06_paper/fits')
PREFIX = ['taskset', '-c', '26-29,70-73', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
          'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src', str(REPO / '.venv/bin/python')]


def worker(args):
    from exp.offline_search.closed_loop import plugin, selftest
    parse = plugin.parse_cli
    plugin.parse_cli = lambda argv: parse([*argv, '--os-no-shadow-native'])
    return selftest.main(args)


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    from exp.offline_search.closed_loop.blind import policy_tail_chunk
    reports = []
    for arm in json.loads((HERE / 'arms_p1.json').read_text()):
        short = 'sp' if arm['suite'] == 'spatial' else 'l10'
        command = PREFIX + [str(HERE / 'plugin_selftests.py'), '--worker', '--blind', '--policy-tail',
                            '--policy-tail-blocks', '1', '--judge', 'guard_only',
                            '--cell', f"groot_{arm['suite']}_cache", '--root', ROOT,
                            '--yaml', f'exp/trace_dual/config/tr_groot_{short}_cache.yaml',
                            '--method', arm['method'], '--kwargs', json.dumps(arm['kwargs']),
                            '--fit-artifact', str(FITS / f"{arm['name']}.pkl"), '--out', str(out / arm['name'])]
        with (out / f"{arm['name']}.log").open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
        report = json.loads((out / arm['name'] / 'selftest_report.json').read_text())
        assert report['PASS']
        eligible = tails = 0
        for path in sorted((out / arm['name'] / 'inputs').glob('*.npz')):
            with np.load(path, allow_pickle=False) as d:
                meta = json.loads(str(d['meta']))
                assert meta['policy_tail_blocks'] == 1 and meta['H'] == 16
                wanted = np.flatnonzero((d['hit'][:-1] == 0) & d['vision'][:-1]) + 1
                used = np.flatnonzero(d['source'] == 'policy_tail')
                assert np.array_equal(wanted, used), (arm['name'], wanted, used)
                for j in used:
                    assert d['a_exec'][j].tobytes() == policy_tail_chunk(d['a_exec'][j - 1], 5).tobytes()
                    assert d['wire_actions'][j].tobytes() == policy_tail_chunk(d['wire_actions'][j - 1], 5).tobytes()
                eligible += len(wanted)
                tails += len(used)
        reports.append(dict(arm=arm['name'], command=command, eligible_misses=eligible, policy_tails=tails,
                            **{k: report[k] for k in ('PASS', 'decisions', 'vision', 'blind', 'miss', 'stage1_calls',
                                                      'broadcasts', 'connections', 'episodes', 'duplicate_rejections',
                                                      'rejected_preflight', 'output_fallbacks', 'partial_looks')}))
        print(json.dumps({k: v for k, v in reports[-1].items() if k != 'command'}), flush=True)
    (HERE / 'results' / 'plugin_selftests.json').write_text(json.dumps(reports, indent=1) + '\n')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        raise SystemExit(worker([x for x in sys.argv[1:] if x != '--worker']))
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args().out)
