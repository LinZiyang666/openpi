"""Run a small DAG of CPU jobs with bounded concurrency; one log per job, a JSON status file at the end.

usage: cpu_jobs.py PLAN.json OUTDIR [--workers N]
PLAN.json: [{"name": ..., "cmd": "...", "after": [names]}]; commands run in /home/weiland/projects/openpi under the
single-thread CPU environment (bitwise fits need OMP=1), niced.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONPATH='.:src')


def main():
    plan, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 16
    out.mkdir(parents=True, exist_ok=True)
    jobs = {j['name']: dict(j, state='waiting') for j in plan}
    running = {}
    while any(j['state'] in ('waiting', 'running') for j in jobs.values()):
        for name, p in list(running.items()):
            if p.poll() is not None:
                jobs[name].update(state='done' if p.returncode == 0 else 'failed', rc=p.returncode, ended=time.strftime('%H:%M:%S'))
                del running[name]
                print(time.strftime('%H:%M:%S'), name, jobs[name]['state'], flush=True)
        for name, j in jobs.items():
            if j['state'] != 'waiting':
                continue
            deps = [jobs[d]['state'] for d in j.get('after', [])]
            if any(s in ('failed', 'skipped') for s in deps):
                j['state'] = 'skipped'
                print(time.strftime('%H:%M:%S'), name, 'skipped', flush=True)
            elif all(s == 'done' for s in deps) and len(running) < workers:
                log = (out/(name+'.log')).open('w')
                running[name] = subprocess.Popen(['nice', '-n', '10', 'bash', '-c', j['cmd']], cwd='/home/weiland/projects/openpi',
                                                 env=ENV, stdout=log, stderr=subprocess.STDOUT)
                j.update(state='running', started=time.strftime('%H:%M:%S'))
        (out/'STATUS.json').write_text(json.dumps(jobs, indent=1))
        time.sleep(5)
    (out/'STATUS.json').write_text(json.dumps(jobs, indent=1))
    bad = [n for n, j in jobs.items() if j['state'] != 'done']
    print('ALL_DONE' if not bad else 'NOT_DONE ' + ' '.join(bad), flush=True)


if __name__ == '__main__':
    main()
