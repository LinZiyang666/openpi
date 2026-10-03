# Adapted from rounds/r06/p3_profiling/dispatch_fence.py; offset ACK/durable receipt semantics retained.
"""Persistent dispatch generations for both opt-in debug transport modes.

Counters are tiny metadata beside (not inside) client payloads and survive verified
telemetry cleanup. No simulator, receiver, scheduling order or RNG changes.
"""
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .protocol import attempt_key


class DispatchFence:
    def __init__(self, directory, journal=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.floor = 0
        # Upgrade safety: an old driver has no dispatch WAL, so its unjournaled
        # generations cannot be enumerated locally. Use a reserved high range
        # for that existing arm, not a guessed max(accepted journal attempts).
        # Fresh arms still start at 1. This marker is never telemetry-cleaned.
        with (self.directory/'bootstrap.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            marker=self.directory/'bootstrap.json'
            if marker.exists():
                state=json.loads(marker.read_text())
            else:
                legacy=journal is not None and Path(journal).exists()
                state=dict(schema='osdebug.dispatch_fence.v1', floor=100000 if legacy else 0,
                           legacy_journal=str(journal) if legacy else None)
                self._write(marker,state)
            self.floor=state['floor']

    @staticmethod
    def _write(path,value):
        fd,tmp=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
        try:
            with os.fdopen(fd,'w') as f:
                json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
            fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)

    def reserve(self,uid,proposed):
        attempt_key(uid,proposed)
        key=hashlib.sha256(uid.encode()).hexdigest()
        with (self.directory/(key+'.lock')).open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.directory/(key+'.json')
            old=json.loads(path.read_text()) if path.exists() else dict(uid=uid,last=self.floor)
            if old['uid']!=uid:
                raise ValueError('dispatch counter uid mismatch')
            n=max(proposed,old['last']+1)
            attempt_key(uid,n)  # fail closed at protocol capacity
            self._write(path,dict(uid=uid,last=n))
            return n


def install(directory,journal=None):
    """Wrap this process's scheduler; keep its stale-result fence consistent."""
    from openpi.conductor.scheduler import EpisodeScheduler
    fence=DispatchFence(directory,journal)
    original=EpisodeScheduler.next_task
    if getattr(original,'_osdebug_dispatch_fence',False):
        raise RuntimeError('dispatch fence already installed')
    def next_task(self,server_key):
        with self._lock:
            task=original(self,server_key)
            if task is None:return None
            n=fence.reserve(task.task_uid,task.attempt)
            self._dispatch_gen[task.task_uid]=n
            return task if n==task.attempt else dataclasses.replace(task,attempt=n)
    next_task._osdebug_dispatch_fence=True
    EpisodeScheduler.next_task=next_task
    return fence
