"""Emit owned arm specs and exact CPU prefit commands."""
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
RUN = Path('/home/weiland/trace_runs/os_closed_loop/r04_k7')
SPEC = 'exp.offline_search.rounds.r04.k7_guard.judge:VisionConfirmedBlindMixedJudge'
PREFIX = ['taskset', '-c', '26-29,70-73', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
          'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src',
          '/home/weiland/projects/openpi/.venv/bin/python']


def main():
    old = json.loads((HERE.parent/'k1_blind/arms_r4.json').read_text())
    names = ['r4b3_p_l10_500_b0g', 'r4b3_p_l10_500_ph2g', 'r4b3_p_l10_500_ph1g',
             'r4b3_p_l10_50_ph2g', 'r4b3_p_sp_500_ph2g']
    arms, commands = [], []
    for name in names:
        row = next(x for x in old if x['name'] == name)
        row['name'] = name.replace('r4b3_', 'r4k7_')
        row['method'] = SPEC
        row['kwargs']['stuck_guard'] = 'vision_confirmed'
        row['cost_ledger'] = True
        row['plugin_args'][-1] = '<RUN>/fits/' + row['name'] + '.pkl'
        row['plugin_args'][:0] = ['--os-root', '/home/weiland/trace_runs/offline_search_store']
        arms.append(row)
        commands.append(PREFIX + ['-m', 'exp.offline_search.closed_loop.plugin', '--os-method', SPEC,
            '--os-kwargs', json.dumps(row['kwargs'], separators=(',', ':')), '--os-cell',
            f"pi05_{row['suite']}_cache", '--os-root', '/home/weiland/trace_runs/offline_search_store',
            '--os-log-dir', str(HERE/'results/prefit'/row['name']), '--os-tag', row['name']]
            + [x.replace('<RUN>', str(RUN)) for x in row['plugin_args'][2:]])
    (HERE/'arms_k7.json').write_text(json.dumps(arms, indent=2)+'\n')
    (HERE/'prefit.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n'
        + '\n'.join(shlex.join(c) for c in commands)+'\n')
    (HERE/'results/prefit_commands.json').write_text(json.dumps(commands, indent=2)+'\n')


if __name__ == '__main__':
    main()
