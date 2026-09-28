"""CPU-only fresh prefits, seven children + this coordinator = eight Python processes."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, PREFIX, RUN, STORE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=RUN)
    p.add_argument('--parallel', type=int, default=7)
    p.add_argument('--resume', action='store_true', help='skip already completed artifacts')
    a = p.parse_args()
    assert 1 <= a.parallel <= 7
    rows = sum([json.loads((HERE / f'arms_{s}.json').read_text()) for s in ('metric', 'trigger_loo')], [])
    jobs = []
    (a.run / 'fits').mkdir(parents=True, exist_ok=True)
    (a.run / 'prefit_logs').mkdir(parents=True, exist_ok=True)
    for r in rows:
        artifact = a.run / 'fits' / (r['name'] + '.pkl')
        if artifact.exists():
            if a.resume:
                continue
            raise FileExistsError(artifact)
        pargs = [x.replace('<RUN>', str(a.run)) for x in r['plugin_args']]
        cmd = PREFIX + ['-m', 'exp.offline_search.closed_loop.plugin', '--os-method', r['method'],
                        '--os-kwargs', json.dumps(r['kwargs']), '--os-cell', f"{r['model']}_{r['suite']}_cache",
                        '--os-log-dir', str(a.run / 'prefit_logs' / r['name']), '--os-tag', r['name']]
        if '--os-root' not in pargs:
            cmd += ['--os-root', STORE]
        jobs.append((r['name'], cmd + pargs))
    (HERE / 'results/prefit_commands.json').write_text(json.dumps([dict(arm=n, argv=c, command=shlex.join(c))
                                                                  for n, c in jobs], indent=1) + '\n')
    running, failures = [], []
    while jobs or running:
        while jobs and len(running) < a.parallel:
            name, cmd = jobs.pop(0)
            log = (a.run / 'prefit_logs' / (name + '.log')).open('w')
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        for item in list(running):
            name, proc, log = item
            if proc.poll() is not None:
                log.close()
                print(name, proc.returncode, flush=True)
                if proc.returncode:
                    failures.append(name)
                running.remove(item)
        if running:
            time.sleep(1)
    assert not failures, failures


if __name__ == '__main__':
    main()
