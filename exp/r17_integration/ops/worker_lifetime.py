"""Recycle owned LIBERO workers only after reporting complete episodes."""
import argparse
import os
import time
from pathlib import Path

from .formal_process import ticks, write_json

RECYCLE_EXIT = 75


def limited_loop(base, maximum, marker, seconds=1800):
    class LimitedLoop(base):
        completed = 0

        def _run_one(self, sock, task):
            super()._run_one(sock, task)
            # The result was sent before stopping, including its attempt fence.
            self.completed += 1

        def run_forever(self, stop=None):
            started = time.monotonic()
            expired = lambda: self.completed > 0 and time.monotonic() - started >= seconds
            super().run_forever(stop=lambda: self.completed >= maximum or expired() or bool(stop and stop()))
            if self.completed >= maximum or expired():
                write_json(marker, dict(schema='r17.worker_recycle.v1', pid=os.getpid(),
                    start_ticks=ticks(os.getpid()), completed_episodes=self.completed,
                    maximum_episodes=maximum, maximum_seconds=seconds, elapsed_seconds=time.monotonic()-started, reason='episode_boundary'))
                raise SystemExit(RECYCLE_EXIT)
    return LimitedLoop


def planned_recycle(entry, code):
    if code != RECYCLE_EXIT or not entry.get('recycle_receipt'):
        return False
    import json
    try:
        marker = json.loads(Path(entry['recycle_receipt']).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (marker.get('schema') == 'r17.worker_recycle.v1' and marker.get('pid') == entry['pid'] and
            marker.get('start_ticks') == entry['start_ticks'] and marker.get('reason') == 'episode_boundary' and
            type(marker.get('maximum_episodes')) is int and marker['maximum_episodes'] > 0 and
            (marker.get('completed_episodes') == marker['maximum_episodes'] or
             0 < marker.get('completed_episodes', 0) <= marker['maximum_episodes'] and
             marker.get('elapsed_seconds', 0) >= marker.get('maximum_seconds', float('inf'))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-episodes', type=int, required=True)
    p.add_argument('--cache-study', action='store_true')
    args, remaining = p.parse_known_args(argv)
    if args.max_episodes < 1:
        p.error('positive episode lifetime required')
    marker = os.environ['R17_WORKER_RECYCLE_RECEIPT']
    from openpi.conductor import worker
    from exp.r17_integration.conductor import worker_entry
    original = worker.WorkerLoop
    # Process-scoped injection through the stock worker interface. No src or
    # simulator edits, task changes, RNG changes, or mid-episode signals.
    worker.WorkerLoop = limited_loop(original, args.max_episodes, marker)
    try:
        if args.cache_study:
            from .cache_study_worker import evidence
            with evidence():
                worker_entry.main(remaining)
        else:
            worker_entry.main(remaining)
    finally:
        worker.WorkerLoop = original


if __name__ == '__main__':
    main()
