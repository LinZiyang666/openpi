"""Bounded retries for explicit tether transport failures, never command errors."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import math
import os
from pathlib import Path
import re
import sys
import subprocess
import threading
import time

RETRY_SECONDS = 300
# A sequential retry must not expire the other node's resident leases.
LEASE_SECONDS = 600
current_journal = ContextVar('r17_tether_retry_journal', default=None)
current_step = ContextVar('r17_tether_retry_step', default=None)


def transient(exc, *, replay_safe=False):
    """Tether's exit64/75 and transport envelope distinguish CLI failures.

    In particular, a remote application exiting1 (even with STALE in its
    traceback), a command timeout, or an ambiguous lost spawn response is not
    permission to execute the application again.
    """
    if getattr(exc, 'r17_retry_exhausted', False):
        return False
    if isinstance(exc, subprocess.TimeoutExpired):
        command = exc.cmd if isinstance(exc.cmd, (list, tuple)) else []
        return replay_safe and bool(command) and Path(command[0]).name=='tether'
    code = getattr(exc, 'returncode', None)
    stderr = getattr(exc, 'stderr', '') or ''
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors='replace')
    text = stderr.lower()
    if code == 75 and 'too_many_in_flight' in text:
        return True
    # The CLI could not dial the broker, so nothing reached the node: always safe to retry
    # (csfw_k3 died on one 'cannot reach broker ... i/o timeout' at 22:12 on 2026-10-13; the next call 13 s later worked).
    if code == 69 and 'cannot reach broker' in text:
        return True
    # Tier-B transfers include "pull (tier B):"; tether uses EX_SOFTWARE
    # (70), as well as 64, for relay/NATS timeouts. Remote application errors
    # and ambiguous spawn replies still must never be replayed.
    envelope = bool(re.search(r'error: (?:exec|pull|push|expose)(?: \(tier [^)]+\))?:', text))
    if any(marker in text for marker in ('remote command exited', 'remote execution deadline',
                                         'traceback (most recent call last)')):
        return False
    rejected = any(marker in text for marker in (
        'node_offline', 'node_unreachable', 'status=stale', 'transport_error',
        'relay_unavailable', 'leader_unavailable', 'agent is offline', 'no recent heartbeat',
        'temporary failure in name resolution'))
    lost_reply = any(marker in text for marker in (
        'transport_timeout', 'connection_lost', 'connection reset by peer',
        'context deadline exceeded', 'unexpected eof', 'nats: timeout',
        'waiting for chunk', 'read pipe: i/o timeout'))
    return code in (64, 70) and envelope and (rejected or replay_safe and lost_reply)


class RetryJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'TETHER_BLIPS.jsonl'
        self.lock = threading.Lock()
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        self.call_path = self.directory/'TETHER_CALLS.jsonl'
        self.calls = [json.loads(line) for line in self.call_path.read_text().splitlines()] if self.call_path.exists() else []

    def record_call(self, operation, node=None):
        row=dict(operation=operation,node=node,monotonic=time.monotonic(),
                 timestamp=datetime.now(timezone.utc).isoformat())
        with self.lock:
            with self.call_path.open('a') as stream:
                stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
            self.calls.append(row)

    def record(self, event):
        with self.lock:
            with self.path.open('a') as stream:
                stream.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.rows.append(event)

    def summary(self):
        with self.lock:
            now=time.monotonic()
            elapsed=max(0,now-self.calls[0]['monotonic']) if self.calls else 0
            return dict(schema='r17.tether_retry.v1', window_seconds=RETRY_SECONDS,
                lease_seconds=LEASE_SECONDS, blips=sum(r['event']=='blip' for r in self.rows),
                events=list(self.rows),call_metrics=dict(total=len(self.calls),
                    observation_seconds=elapsed,
                    average_calls_per_minute=len(self.calls)*60/elapsed if elapsed else None,
                    last_minute=sum(r['monotonic']>=now-60 for r in self.calls),
                    by_node={str(node):sum(r['node']==node for r in self.calls) for node in {r['node'] for r in self.calls}}))


@contextmanager
def retry_scope(directory):
    existing = current_journal.get()
    journal = existing if existing and existing.directory == Path(directory) else RetryJournal(directory)
    token = current_journal.set(journal)
    try:
        yield journal
    finally:
        from .formal_process import write_json
        write_json(journal.directory / 'TETHER_RETRY_RECEIPT.json', journal.summary())
        current_journal.reset(token)


def campaign_retries(function):
    @wraps(function)
    def wrapped(campaign, directory, *args, **kwargs):
        with retry_scope(directory):
            return function(campaign, directory, *args, **kwargs)
    return wrapped


def retry(operation, *, label, node=None, seconds=RETRY_SECONDS, sleeper=None, clock=None, on_event=None,
          replay_safe=False):
    """Retry an explicit transport rejection for at most seconds after failure.

    Command execution timeouts stay real errors. The retry window bounds the
    waits; a final in-flight CLI call retains its own execution timeout.
    """
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('retry window must be finite and positive')
    sleeper, clock = sleeper or time.sleep, clock or time.monotonic
    journal = current_journal.get()
    started, attempt = None, 0
    def emit(kind, **fields):
        event = dict(event=kind, operation=label, node=node, attempt=attempt,
            timestamp=datetime.now(timezone.utc).isoformat(), context=current_step.get(), **fields)
        if journal:
            journal.record(event)
        if on_event:
            on_event(event)
        print('R17 tether '+json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)
    while True:
        attempt += 1
        try:
            result = operation()
        except Exception as exc:
            if not transient(exc, replay_safe=replay_safe):
                raise
            now = clock()
            if started is None:
                started = now
            remaining = seconds-(now-started)
            delay = min(2**min(attempt, 5), 30, max(0, remaining))
            stderr = getattr(exc, 'stderr', '')
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors='replace')
            emit('blip', error=repr(exc), stderr=stderr, returncode=getattr(exc,'returncode',None),
                elapsed_seconds=now-started, retry_delay_seconds=delay)
            if remaining > 0:
                sleeper(delay)
            if remaining <= 0 or clock()-started >= seconds:
                emit('expired', elapsed_seconds=clock()-started, stderr=stderr)
                exc.r17_retry_exhausted = True
                raise
        else:
            if started is not None:
                emit('recovered', elapsed_seconds=clock()-started)
            return result
