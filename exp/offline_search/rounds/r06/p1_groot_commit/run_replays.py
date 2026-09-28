"""Run the replay matrix (replay.py jobs, <= 8 concurrent processes on CPUs 26-29,70-73).

Groups (every job replays ALL 500 recorded episodes of its replay cell through the installed plugin):
  N  nesting A vs B-off: both models x 4 cells, cache replay cell. A = the deployed A arm (spec + its original fit
     artifact); B-off = the deployed B class/kwargs with guards=False, fitted fresh (pi05 B = C10 CommitJudge,
     GR00T B = P1 GrootCommitJudge).
  P  pi05 decision-logic parity: C10 (deployed r05_q1 fit) vs P1 GrootCommitJudge fitted fresh on pi05, 4 cells x
     {cache, inf} replay cells.
  G  GR00T B: the four deployed r06_paper fits x {cache, inf} replay cells.
  T  test-only control for N: A with the G3 wrapper's k-th-slot tie rule (tie_rule.StableTieBlindAWM), fitted fresh.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(HERE))
from make_arms import source_rows, kwargs as b_kwargs  # noqa: E402

RUNS = Path('/home/weiland/trace_runs/os_closed_loop')
P1 = 'exp.offline_search.rounds.r06.p1_groot_commit.judge:GrootCommitJudge'
C10 = 'exp.offline_search.rounds.r05.q1_commit.judge:CommitJudge'
TIE = 'exp.offline_search.rounds.r06.p1_groot_commit.tie_rule:StableTieBlindAWM'
ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
           PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='.:src')
PREFIX = ['taskset', '-c', '26-29,70-73', str(REPO / '.venv/bin/python'), str(HERE / 'replay.py')]
CELLS = ('l10_50', 'l10_500', 'sp_50', 'sp_500')
SUITE = {'l10': 'l10', 'sp': 'spatial'}
B_FLAGS = ['--judge', 'guard_only', '--policy-tail', '--blocks', '1']


def fit_of(row):
    args = row['plugin_args']
    path = args[args.index('--os-fit-artifact') + 1]
    return path.replace('<RUN>', str(RUNS / 'r06_paper'))


def jobs():
    src = source_rows()
    out = []
    for model in ('pi05', 'groot'):
        for cell in CELLS:
            suite, scale = SUITE[cell.split('_')[0]], int(cell.split('_')[1])
            fcell = f'{model}_{suite}_cache'
            a = src[(model, 'A', cell)]['row']
            out.append(dict(name=f'N_{model}_{cell}_A', method=a['method'], kwargs=a['kwargs'], cell=fcell,
                            replay=fcell, fit=fit_of(a), flags=[]))
            b = src[(model, 'B', cell)]['row']
            kw = copy.deepcopy(b['kwargs'])
            kw['guards'] = False
            out.append(dict(name=f'N_{model}_{cell}_Boff', method=b['method'], kwargs=kw, cell=fcell, replay=fcell,
                            fit='', flags=B_FLAGS))
            # Test-only control: A with the G3 wrapper's k-th-slot tie rule (tie_rule.py), fitted fresh.
            out.append(dict(name=f'T_{model}_{cell}_Astable', method=TIE, kwargs=a['kwargs'], cell=fcell,
                            replay=fcell, fit='', flags=[]))
    for cell in CELLS:
        suite, scale = SUITE[cell.split('_')[0]], int(cell.split('_')[1])
        fcell = f'pi05_{suite}_cache'
        b = src[('pi05', 'B', cell)]['row']
        assert b['method'] == C10 and b['kwargs'] == b_kwargs(scale)
        for arm in ('cache', 'inf'):
            out.append(dict(name=f'P_pi05_{cell}_{arm}_C10', method=C10, kwargs=b['kwargs'], cell=fcell,
                            replay=f'pi05_{suite}_{arm}', fit=fit_of(b), flags=B_FLAGS))
            out.append(dict(name=f'P_pi05_{cell}_{arm}_P1', method=P1, kwargs=b_kwargs(scale), cell=fcell,
                            replay=f'pi05_{suite}_{arm}', fit='', flags=B_FLAGS))
    for cell in CELLS:
        suite = SUITE[cell.split('_')[0]]
        b = src[('groot', 'B', cell)]['row']
        for arm in ('cache', 'inf'):
            out.append(dict(name=f'G_groot_{cell}_{arm}', method=b['method'], kwargs=b['kwargs'],
                            cell=f'groot_{suite}_cache', replay=f'groot_{suite}_{arm}', fit=fit_of(b), flags=B_FLAGS))
    return out


def command(job, root, episodes):
    cmd = PREFIX + ['--method', job['method'], '--kwargs', json.dumps(job['kwargs']), '--cell', job['cell'],
                    '--replay-cell', job['replay'], '--episodes', episodes, '--tag', job['name'],
                    '--out', str(root / job['name'])] + job['flags']
    if job['fit']:
        cmd += ['--fit-artifact', job['fit']]
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--groups', default='NPG')
    p.add_argument('--episodes', default='all')
    p.add_argument('--parallel', type=int, default=7)
    a = p.parse_args()
    assert a.parallel <= 7  # + this driver process = 8 processes
    a.root.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs() if j['name'][0] in a.groups]
    # Biggest (l10, 500-library) jobs first.
    todo.sort(key=lambda j: (('l10' not in j['name']), ('_500' not in j['name']), j['name']))
    (a.root / f'jobs_{a.groups}.json').write_text(json.dumps([dict(**j, command=command(j, a.root, a.episodes))
                                                            for j in todo], indent=1))
    running, done = {}, []
    env = {**os.environ, **ENV}
    while todo or running:
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            log = open(a.root / f"{j['name']}.log", 'w')
            running[j['name']] = (subprocess.Popen(command(j, a.root, a.episodes), cwd=REPO, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT), log, time.time())
        time.sleep(2)
        for name, (proc, log, t0) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                done.append(dict(name=name, rc=proc.returncode, wall_s=round(time.time() - t0, 1)))
                print(json.dumps(done[-1]), flush=True)
                del running[name]
    previous = json.loads((a.root / 'done.json').read_text()) if (a.root / 'done.json').exists() else []
    names = {d['name'] for d in done}
    done = [d for d in previous if d['name'] not in names] + done      # later group runs extend the record
    (a.root / 'done.json').write_text(json.dumps(done, indent=1))
    bad = [d for d in done if d['rc']]
    print('FAILED' if bad else 'ALL OK', bad)
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
