"""All-vision (B=0, K7's stock branch) guard trigger counts on every recorded episode of the store query cells.

For each deployed B fit (GR00T P1 r06_paper, pi05 C10 r05_q1) and each recorded cell ({cache, inf}), every decision
is queried with the recorded QueryView (recorded executed history, no blind gaps): this is the regime K7 routes to
stock MixedJudge.query and the only place where P1's terminal-sign correction acts. Counts per guard bit, per
os_reason, and terminal-row gripper states with both signs. Fixed recorded inputs; hypothetical verdicts do not
alter the stream (same protocol as K7's rates table).
"""
import argparse
import collections
import json
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = '/home/weiland/trace_runs/offline_search_store'
FITS = {'groot': Path('/home/weiland/trace_runs/os_closed_loop/r06_paper/fits') / 'r6p1_c10_g_{s}_{n}.pkl',
        'pi05': Path('/home/weiland/trace_runs/os_closed_loop/r05_q1/fits') / 'r5q1_c10_p_{s}_{n}.pkl'}
SHORT = {'l10': 'l10', 'spatial': 'sp'}
BITS = ('stuck', 'terminal_closed', 'overtime', 'no_progress', 'dispersion', 'grip')
PREFIX = ['taskset', '-c', '26-29,70-73', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
          'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src', str(REPO / '.venv/bin/python')]


def one(model, suite, scale, arm, out):
    from exp.offline_search.harness import api, store
    from exp.offline_search.rounds.r04.k1_blind.checks import view
    path = str(FITS[model]).format(s=SHORT[suite], n=scale)
    with open(path, 'rb') as f:
        m = pickle.load(f)['method']
    sign = -1. if model == 'groot' else 1.
    qc = store.QueryCell(ROOT, f'{model}_{suite}_{arm}')
    A = api.QueryArrays(qc)
    bits, reasons, c = collections.Counter(), collections.Counter(), collections.Counter()
    for e in qc.episodes:
        m.reset(view(qc, A, e['start']).episode)
        for i in range(e['start'], e['end']):
            ex = m.query(view(qc, A, i)).extras
            flags = int(ex['os_flags'])
            for b, name in enumerate(BITS):
                if flags >> b & 1:
                    bits[name] += 1
            if ex['os_force_miss'] == 1.:
                reasons[BITS[int(ex['os_reason']) - 1]] += 1
            c['decisions'] += 1
            if ex.get('term1') == 1.:
                c['terminal_rows'] += 1
                c['terminal_closed_model_sign'] += int(ex.get('gexec', 0.) * sign > 0)
                c['terminal_closed_opposite_sign'] += int(ex.get('gexec', 0.) * sign < 0)
    r = dict(model=model, suite=suite, scale=scale, cell=f'{model}_{suite}_{arm}', fit=path,
             episodes=len(qc.episodes), **c, force_miss=sum(reasons.values()),
             force_miss_share=sum(reasons.values()) / c['decisions'], flag_bits=dict(bits), reasons=dict(reasons))
    Path(out).write_text(json.dumps(r, indent=1) + '\n')
    print(json.dumps(r), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--one', nargs=5)
    p.add_argument('--work', type=Path, default=Path('/tmp/p1_allvision'))
    p.add_argument('--parallel', type=int, default=7)
    a = p.parse_args()
    if a.one:
        model, suite, scale, arm, out = a.one
        return one(model, suite, int(scale), arm, out)
    assert a.parallel <= 7
    a.work.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s, n, arm) for m in ('groot', 'pi05') for s in ('l10', 'spatial') for n in (500, 50)
            for arm in ('cache', 'inf')]
    running, results = [], []
    while jobs or running:
        while jobs and len(running) < a.parallel:
            m, s, n, arm = jobs.pop(0)
            out = a.work / f'{m}_{s}_{n}_{arm}.json'
            log = open(a.work / f'{m}_{s}_{n}_{arm}.log', 'w')
            running.append((subprocess.Popen(PREFIX + [str(Path(__file__).resolve()), '--one', m, s, str(n), arm,
                                                       str(out)], cwd=REPO,
                                             stdout=log, stderr=subprocess.STDOUT), out, log))
        for item in list(running):
            proc, out, log = item
            if proc.poll() is not None:
                log.close()
                assert proc.returncode == 0, out
                results.append(json.loads(out.read_text()))
                running.remove(item)
        if running:
            import time
            time.sleep(2)
    results.sort(key=lambda r: (r['model'], r['suite'], r['scale'], r['cell']))
    (HERE / 'results' / 'allvision_rates.json').write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
