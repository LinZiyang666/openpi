"""A100-side: run preregistered extension blocks right behind the current GPU
holder (blocking exclusive lock, so no idle gap), then pair them.

Runs are executed in the given fixed order; the queue stops on infrastructure
failure and never reruns an accepted episode. Analysis pairs each block's cold
starts with the formal reference and with each other (same-card).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/exouser/xgpu_1017/cl18')


def analyse(model, names):
    from paired_sr import paired_interval
    report = []
    for name in names:
        base = ROOT / model / name
        reference = json.loads((base / 'REFERENCE.json').read_text())['outcomes']
        uids = json.loads((base / 'UIDS.json').read_text())
        runs = {}
        for cold in (1, 2):
            r = ROOT / 'runs' / f'{model}_{name}_cold{cold}'
            if not (r / 'RUN.json').exists():
                continue
            receipt = json.loads((r / 'RUN.json').read_text())
            if receipt['status'] != 'COMPLETE' or not receipt['mps_cleanup']['daemon_closed']:
                continue
            out = json.loads((r / 'OUTCOMES.json').read_text())
            if set(out) != set(uids):
                raise ValueError('missing accepted original UID in ' + str(r))
            runs[cold] = [int(out[u]['success']) for u in uids]
        formal = [int(reference[u]['success']) for u in uids]
        row = dict(model=model, arm=name, episodes=len(uids), colds=sorted(runs))
        for cold, v in runs.items():
            row[f'a100_cold{cold}_vs_formal'] = paired_interval(v, formal)
        if len(runs) == 2:
            row['same_card'] = paired_interval(runs[1], runs[2])
        row['per_episode'] = [dict(uid=u, formal=formal[i], **{f'cold{c}': runs[c][i] for c in runs})
                              for i, u in enumerate(uids)]
        report.append(row)
    target = ROOT / 'analysis'
    target.mkdir(exist_ok=True)
    (target / f'{model}_EXT_{"_".join(names)}_PAIRED.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in r.items() if k != 'per_episode'} for r in report]), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--runs', nargs='+', required=True, help='name:cold in fixed order')
    a = p.parse_args()
    plan = [(r.split(':')[0], int(r.split(':')[1])) for r in a.runs]
    receipt = ROOT / 'queue' / f'{a.model}_ext_{a.tag}.json'
    if receipt.exists():
        raise ValueError('extension queue already submitted; never rerun accepted episodes')
    waiting = ROOT / 'queue' / f'{a.model}_ext_{a.tag}.waiting'
    waiting.write_text(json.dumps(dict(pid=os.getpid(), since=time.time(), runs=a.runs)) + '\n')
    lock = (ROOT / 'exclusive_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    waiting.unlink()
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True)
    if 'A100' not in names or 'H100' in names:
        raise ValueError('dedicated A100 only')
    state = dict(model=a.model, tag=a.tag, status='RUNNING', pid=os.getpid(), started=time.time(), plan=a.runs, completed=[])

    def save():
        receipt.write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        for name, cold in plan:
            cmd = [sys.executable, str(Path(__file__).with_name('a100_closedloop_run_1018.py')), 'run',
                   '--model', a.model, '--arm', name, '--cold', str(cold)]
            state['current'] = dict(arm=name, cold=cold)
            save()
            subprocess.run(cmd, check=True)
            run = json.loads((ROOT / 'runs' / f'{a.model}_{name}_cold{cold}' / 'RUN.json').read_text())
            if run['status'] != 'COMPLETE' or not run['mps_cleanup']['daemon_closed']:
                raise RuntimeError('run did not close cleanly')
            state['completed'].append(dict(arm=name, cold=cold))
            save()
        analyse(a.model, sorted({n for n, _ in plan}))
        state['status'] = 'COMPLETE'
    except BaseException as e:
        state.update(status='FAILED', error=repr(e))
        raise
    finally:
        state['ended'] = time.time()
        save()


if __name__ == '__main__':
    main()
