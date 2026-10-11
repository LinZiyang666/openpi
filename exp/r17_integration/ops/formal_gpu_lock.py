"""Host-local GPU exclusion, held by the launcher's expiring remote lease.

Invoked on H100 only for library campaigns. Reuses Q's lock convention, never
opens the coordinator/4090 lock. No CUDA, model or process census is involved.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import time

from .formal_process import write_json

LOCK_PATH = '/tmp/r17_gpu.lock'


@contextmanager
def exclusive(path=LOCK_PATH):
    with Path(path).open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            pass
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def hold(ready, campaign):
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    with exclusive():
        write_json(ready,dict(status='PASS',campaign=campaign,pid=os.getpid(),lock=LOCK_PATH))
        while not stopped:
            time.sleep(.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready',required=True)
    parser.add_argument('--campaign',required=True)
    args = parser.parse_args()
    hold(args.ready,args.campaign)


if __name__=='__main__':
    main()
