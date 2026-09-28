"""Bounded subprocess runner; commands always include the CPU/thread/device prefix."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from exp.offline_search.rounds.r06.p2_ablations.make_arms import PREFIX


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jobs', type=Path, help='JSON [{name, args: [python args...]}]')
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--parallel', type=int, default=7)
    a = p.parse_args()
    assert 1 <= a.parallel <= 7
    a.logs.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(a.jobs.read_text())
    running, failed = [], []
    while jobs or running:
        while jobs and len(running) < a.parallel:
            job = jobs.pop(0)
            f = (a.logs / (job['name'] + '.log')).open('w')
            proc = subprocess.Popen(PREFIX + job['args'], stdout=f, stderr=subprocess.STDOUT)
            running.append((job['name'], proc, f))
        for item in list(running):
            name, proc, f = item
            if proc.poll() is not None:
                f.close()
                print(name, proc.returncode, flush=True)
                if proc.returncode:
                    failed.append(name)
                running.remove(item)
        if running:
            time.sleep(1)
    assert not failed, failed


if __name__ == '__main__':
    main()
