"""Emit six exact arm specs and CPU-only prefit commands; never launch a server."""
import json
import os
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
PREFIX = ['taskset', '-c', '26-29,70-73', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src',
          '/home/weiland/projects/openpi/.venv/bin/python']
ROOT = '/home/weiland/trace_runs/offline_search_store'


def write(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    arms, commands, actual = [], [], []
    for family, suite, scale in [(f, s, n) for f in ('c10', 'd1')
                                  for s in (('l10', 'spatial') if f == 'c10' else ('l10',))
                                  for n in (50, 500)]:
        name = f'r5q1_{family}_p_{"sp" if suite == "spatial" else suite}_{scale}'
        method = 'exp.offline_search.rounds.r05.q1_commit.judge:' + ('CommitJudge' if family == 'c10' else 'GraspCheckJudge')
        kw = dict(base_kwargs=dict(lib='current' if scale == 50 else 'big', kref=5 if scale == 50 else 8,
                                   serving='anchor_tail', budget=1, gates='budget_only'),
                  progress_guard='noprog_span', events='none', stuck_guard='vision_confirmed')
        if family == 'c10':
            kw.update(policy_tail_gate='lifecycle', monitor='off')
        flags = ['--os-root', ROOT, '--os-blind']
        if family == 'c10':
            flags.append('--os-policy-tail')
        flags += ['--os-judge', 'guard_only', '--os-no-shadow-native', '--os-fit-artifact', f'<RUN>/fits/{name}.pkl']
        arms.append(dict(name=name, model='pi05', suite=suite, mode='plugin', method=method, kwargs=kw,
                         full_model=True, cost_ledger=True, client_overrides=dict(replan_steps=5), plugin_args=flags))
        cmd = PREFIX + ['-m', 'exp.offline_search.closed_loop.plugin', '--os-method', method,
                        '--os-kwargs', json.dumps(kw, separators=(',', ':')), '--os-cell', f'pi05_{suite}_cache',
                        '--os-log-dir', f'<RUN>/prefit_logs/{name}', '--os-tag', name] + flags
        commands.append(cmd)
        actual.append([v.replace('<RUN>/fits', '/tmp/q1_fits').replace('<RUN>/prefit_logs', '/tmp/q1_prefit_logs') for v in cmd])
    write(HERE / 'arms_q1.json', json.dumps(arms, indent=2) + '\n')
    write(HERE / 'prefit_commands.json', json.dumps(dict(template=commands, executed=actual), indent=2) + '\n')
    write(HERE / 'prefit.sh', '#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n'
          + '\n'.join(shlex.join(cmd) for cmd in actual) + '\n')


if __name__ == '__main__':
    main()
